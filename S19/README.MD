# NanoGPT Deployment with Gradio on Hugging Face

This repository contains the code and instructions for training a NanoGPT model, inspired by Andrej Karpathy's tutorials, and deploying it using a Gradio app on Hugging Face.

## Table of Contents
- [Introduction](#introduction)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction

NanoGPT is a lightweight implementation of GPT (Generative Pre-trained Transformer) for educational purposes. This project demonstrates how to train NanoGPT and deploy it using a Gradio app on Hugging Face.

## Model Training

The NanoGPT model was trained following the guidelines and code provided by Andrej Karpathy. The training process involved:

1. Setting up the training environment.
2. Preprocessing the dataset.
3. Configuring the model and training parameters.
4. Training the model on a GPU-enabled environment.

## Deployment

After training, the model was deployed using Gradio to create an interactive web application. This Gradio app was then hosted on Hugging Face, allowing users to interact with the model directly from their browsers.

## How to Use

You can interact with the deployed model by visiting the following [Hugging Face link](your-hugging-face-link).

To run the project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/nanogpt-gradio-deployment.git
    cd nanogpt-gradio-deployment
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Gradio app:
    ```bash
    python app.py
    ```

## Dependencies

The project relies on the following main dependencies:
- `transformers`
- `torch`
- `gradio`
- `huggingface_hub`

For a complete list, refer to the `requirements.txt` file.

## Usage

The primary script for running the Gradio app is `app.py`. This script loads the trained NanoGPT model and sets up the Gradio interface for text generation.

```python
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model_name = "your-model-name"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
iface.launch()
