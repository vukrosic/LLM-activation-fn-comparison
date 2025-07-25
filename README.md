# Activation Function Comparison in Transformer Models

This project investigates the effects of different activation functions and the presence of attention bias in a small GPT-style transformer.

## 🔬 Experiment Overview

We compare the training dynamics and performance of three activation functions:

* **ReLU**
* **GELU**
* **SiLU**

Each was tested with and without attention bias in the transformer architecture.

### 🔧 Setup

* **Model:** GPT-style transformer

  * 6 layers, 8 heads, 384 hidden dimension
* **Dataset:** Subset of *Cosmopedia-v2* (500,000 tokens)
* **Training:**

  * 5,000 steps
  * Batch size: 12
  * Gradient accumulation: 4
* **Optimizer:** AdamW (lr = 1e-4, weight decay = 0.1)
* **Variants tested:** All combinations of `{ReLU, GELU, SiLU} × {Bias=True, Bias=False}`

## 📊 Results

Training and validation metrics were tracked for each variant:

* **Training Loss**
* **Validation Loss**
* **Validation Accuracy**
* **Validation Perplexity**

<p align="center">
  <img src="./experiment_images/train_loss_comparison.png" width="400"/>
  <img src="./experiment_images/val_loss_comparison.png" width="400"/>
  <br/>
  <img src="./experiment_images/val_accuracy_comparison.png" width="400"/>
  <img src="./experiment_images/val_perplexity_comparison.png" width="400"/>
</p>

## 📈 Observations

While **GELU with attention bias** achieved the best final validation loss (4.7966), all activation functions performed similarly overall. There is no significant advantage for any single configuration based on this dataset and setup. Minor differences may reflect noise or dataset variance.

## 📌 Conclusion

There is no clearly superior activation function in this setting. For small transformer models on moderate data, **activation choice and attention bias may not drastically affect performance**—but deeper analysis or larger-scale experiments may reveal more nuanced behavior.