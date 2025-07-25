import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import warnings
import os
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    activation: str = 'silu'  # Added for experiment
    use_attention_bias: bool = False # Added for experiment

    # Training parameters
    batch_size: int = 12
    max_steps: int = 5000 # Changed from 5000
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 100 # Changed from 500
    eval_steps: int = 50

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    if os.path.exists(cache_file):
        # print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size
        # print(f"✅ Loaded {len(tokens):,} tokens from cache")
        return tokenizer, tokens

    # print(f"🔄 Processing new data (will cache for future use)")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)
    
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    # print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    cached_data = {'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    # print(f"💾 Cached data to {cache_file}")
    return tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, use_attention_bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=use_attention_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=use_attention_bias)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        Q = self.rotary(Q)
        K = self.rotary(K)
        attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'silu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if activation.lower() == 'relu':
            self.activation = F.relu
        elif activation.lower() == 'gelu':
            self.activation = F.gelu
        elif activation.lower() == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1, activation: str = 'silu', use_attention_bias: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, use_attention_bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout, config.activation, config.use_attention_bias)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.output_dropout(x)
        return self.lm_head(x)

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    model.eval()
    total_loss, total_tokens, total_correct = 0, 0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps: break
            x, y = x.to(device), y.to(device)
            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    print(f"\n🚀 Training with activation: {config.activation}")
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinimalLLM(config).to(device)
    
    # print(f"  📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    warmup_steps = config.max_steps // 20
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler() if config.use_amp else None
    
    history = {'steps': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_perplexity': []}
    
    model.train()
    step = 0
    # pbar = tqdm(total=config.max_steps, desc=f"Training ({config.activation})")
    
    while step < config.max_steps:
        for x, y in train_loader:
            if step >= config.max_steps: break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()

            if step % config.eval_every == 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                history['steps'].append(step)
                history['train_loss'].append(loss.item() * config.gradient_accumulation_steps)
                history['val_loss'].append(eval_metrics['val_loss'])
                history['val_accuracy'].append(eval_metrics['val_accuracy'])
                history['val_perplexity'].append(eval_metrics['val_perplexity'])
                # pbar.set_postfix({{
                #     'loss': f"{{history['train_loss'][-1]:.3f}}",
                #     'val_loss': f"{{eval_metrics['val_loss']:.3f}}",
                #     'val_ppl': f"{{eval_metrics['val_perplexity']:.2f}}"
                # }})

            step += 1
            # pbar.update(1)

    # pbar.close()
    return history

def plot_results(results: Dict[str, Dict], output_dir: str = "experiment_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['train_loss', 'val_loss', 'val_accuracy', 'val_perplexity']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for activation, history in results.items():
            label = activation.replace("_bias_True", "_Bias").replace("_bias_False", "_NoBias").upper()
            plt.plot(history['steps'], history[metric], label=label, alpha=0.8)
        
        plt.title(f'{metric.replace("_", " ").title()} vs. Training Steps')
        plt.xlabel("Steps")
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        # print(f"📈 Saved plot to {plot_path}")

def generate_report(results: Dict[str, Dict], output_file: str = "report.md"):
    report_content = """
# Experiment Report: Activation Function Comparison

This report details an experiment comparing the performance of three different activation functions (ReLU, GELU, and SiLU) and the effect of attention bias within the feed-forward network and attention mechanism of a small transformer model.

## Experimental Setup

- **Model:** A minimal GPT-style model with 6 layers, 8 attention heads, and a model dimension of 384.
- **Dataset:** A subset of the Cosmopedia-v2 dataset, tokenized to a maximum of 500,000 tokens.
- **Training:** Each model variant was trained for 1000 steps with a batch size of 24 and gradient accumulation of 4.
- **Optimizer:** AdamW with a learning rate of 1e-4 and weight decay of 0.1.
- **Variable:** The activation function in the FFN and the presence of attention bias were changed for each run.

## Results

The following plots compare the training and validation metrics for each activation function across the training process.

### Training Loss Comparison

![Training Loss](./experiment_images/train_loss_comparison.png)

### Validation Loss Comparison

![Validation Loss](./experiment_images/val_loss_comparison.png)

### Validation Accuracy Comparison

![Validation Accuracy](./experiment_images/val_accuracy_comparison.png)

### Validation Perplexity Comparison

![Validation Perplexity](./experiment_images/val_perplexity_comparison.png)

## Conclusion
"""
    # Find best performer
    best_combination = None
    best_final_loss = float('inf')
    for combination, history in results.items():
        final_loss = history['val_loss'][-1]
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_combination = combination

    best_combination_formatted = best_combination.replace("_", " ").upper()
    report_content += f"Based on the final validation loss, **{best_combination_formatted}** performed the best, achieving a validation loss of {best_final_loss:.4f}. "
    report_content += "The experiments show that both activation functions and the presence of attention bias can influence model performance. Further analysis of the plots is recommended to understand the specific trade-offs and learning dynamics for each combination."

    with open(output_file, 'w') as f:
        f.write(report_content)
    # print(f"📄 Generated report at {output_file}")


if __name__ == "__main__":
    # print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    # if torch.cuda.is_available():
    #     print(f"GPU: {torch.cuda.get_device_name()}")

    base_config = ModelConfig()
    
    tokenizer, tokens = load_and_cache_data(base_config)
    dataset = TextTokenDataset(tokens, base_config.max_seq_len)
    
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    experiment_results = {}
    activation_functions = ['relu', 'gelu', 'silu']
    attention_bias_settings = [True, False]

    for activation in activation_functions:
        for use_attention_bias in attention_bias_settings:
            print(f"\n🚀 Training with activation: {activation}, attention bias: {use_attention_bias}")
            config = ModelConfig(activation=activation, vocab_size=base_config.vocab_size, use_attention_bias=use_attention_bias)
            history = train_model(config, train_loader, val_loader)
            experiment_results[f"{activation}_bias_{use_attention_bias}"] = history
    
    plot_results(experiment_results)
    generate_report(experiment_results, output_file="report.md")

    # print(f"\n🎉 EXPERIMENT COMPLETED!")
