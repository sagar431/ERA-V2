# GPT-2 Paper Implementation with Training and Deployment

## Overview
This repository contains a PyTorch implementation of a GPT-2 model. The implementation is based on the tutorial videos by Andrew Karpathy. The model includes a CausalSelfAttention module, an MLP module, and various utilities for training and deployment. The model has been trained for 5000 steps and deployed using a Gradio app on Hugging Face.

## Table of Contents
1. [Model Architecture](#model-architecture)
2. [Training](#training)
3. [Deployment](#deployment)
4. [Usage](#usage)
5. [Requirements](#requirements)
6. [Acknowledgements](#acknowledgements)

## Model Architecture
The model is based on the architecture described in the GPT-2 paper, including the following components:
- **CausalSelfAttention**: Implements the self-attention mechanism with causal masking.
- **MLP**: A feed-forward neural network with GELU activation.
- **Block**: A transformer block that includes layer normalization, self-attention, and MLP.
- **GPT**: The complete GPT model with embedding layers, transformer blocks, and a language modeling head.

### CausalSelfAttention
```python
class CausalSelfAttention(nn.Module):
    ...
```

### MLP
```python
class MLP(nn.Module):
    ...
```

### Block
```python
class Block(nn.Module):
    ...
```

### GPT
```python
class GPT(nn.Module):
    ...
```

## Training
The training script initializes the model, sets up the optimizer, and trains the model on input data.

### Training Script
```python
# Define training parameters
max_lr = 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

# Learning rate scheduler
def get_lr(it):
    ...
```

### Data Loading
The `DataLoaderLite` class handles loading and batching the data.

```python
class DataLoaderLite:
    ...
```

### Training Loop
```python
# Initialize model and optimizer
model = GPT(GPTConfig())
model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

# Training loop
for step in range(max_steps):
    ...
```

## Deployment
The model is deployed using a Gradio app, which provides an interactive interface for text generation.

### Gradio App
```python
import gradio as gr

def generate_text(prompt):
    ...
    
interface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
interface.launch()
```

## Usage
To use the model for text generation, simply run the Gradio app and provide a text prompt.

### Example
```bash
python app.py
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Gradio
- tiktoken

Install the required packages using pip:
```bash
pip install torch transformers gradio tiktoken
```

## Acknowledgements
This implementation is inspired by the GPT-2 paper and various open-source repositories. Special thanks to Andrew Karpathy for his detailed tutorial videos that guided this implementation. Also, thanks to the contributors of the minGPT project for their efficient codebase.

---

This README provides a comprehensive guide to understanding, training, and deploying the GPT-2 model. For further details, refer to the source code and comments within the scripts.