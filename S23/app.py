import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation

# Load the CLIPSeg model and processor
processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def segment_image(input_image, text_prompt):
    # Preprocess the image
    inputs = processor(text=[text_prompt], images=[input_image], padding="max_length", return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted segmentation
    preds = outputs.logits.squeeze().sigmoid()

    # Convert the prediction to a PIL image
    segmentation = (preds > 0.5).float()
    segmentation_image = Image.fromarray((segmentation.numpy() * 255).astype(np.uint8))

    return segmentation_image

# Create Gradio interface
iface = gr.Interface(
    fn=segment_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Text Prompt", placeholder="Enter a description of what to segment...")
    ],
    outputs=gr.Image(type="pil", label="Segmentation Result"),
    title="CLIPSeg Image Segmentation",
    description="Upload an image and provide a text prompt to segment objects.",
    examples=[
        ["path/to/example_image1.jpg", "car"],
        ["path/to/example_image2.jpg", "person"],
    ]
)

# Launch the interface
iface.launch()