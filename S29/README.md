# Fine-tuned Phi-2 Chatbot

This project features a chatbot powered by a fine-tuned version of the Phi-2 language model. The chatbot is implemented using Gradio and optimized for CPU inference, making it suitable for deployment on platforms like Hugging Face Spaces.

## Project Overview

- **Model**: Fine-tuned Phi-2 (Microsoft)
- **Fine-tuning Dataset**: OpenAssistant Conversations Dataset (OASST1)
- **Interface**: Gradio Chat Interface
- **Deployment**: Optimized for Hugging Face Spaces (CPU)

## Features

- Utilizes a fine-tuned version of the Phi-2 model
- Implements dynamic quantization for improved CPU performance
- Provides a user-friendly chat interface via Gradio
- Optimized for deployment on CPU-based environments

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/phi2-finetune-chatbot.git
   cd phi2-finetune-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the Gradio app locally:

1. Run the following command:
   ```
   python app.py
   ```

2. Open your web browser and navigate to the URL displayed in the console (typically `http://127.0.0.1:7860`).

3. Interact with the chatbot by typing messages in the input box and pressing Enter.

## Deployment on Hugging Face Spaces

This project is designed to be easily deployed on Hugging Face Spaces:

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose "Gradio" as the SDK.
3. Upload the `app.py` and `requirements.txt` files to your Space.
4. Set the Python version to 3.10.13 in your Space settings.
5. Choose the CPU hardware option.

The Space will automatically build and deploy your app.

## Model Information

- **Base Model**: [Microsoft's Phi-2](https://huggingface.co/microsoft/phi-2)
- **Fine-tuned Model**: [sagar007/phi2_finetune](https://huggingface.co/sagar007/phi2_finetune)
- **Quantization**: Dynamic quantization is applied for CPU optimization

## Limitations

- The model runs on CPU, which may result in slower inference times compared to GPU execution.
- The quality of responses may vary based on the fine-tuning dataset and the limitations of the base Phi-2 model.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit issues or pull requests.

## License

[Specify your license here, e.g., MIT License]

## Acknowledgments

- Microsoft for the Phi-2 model
- Hugging Face for the model hosting and Spaces platform
- The creators of the OpenAssistant Conversations Dataset