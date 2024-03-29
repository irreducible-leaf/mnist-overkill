import os
import gradio as gr
import torch
import numpy as np
from PIL import Image
from model import ModelArchitecture


# Define the function to load the model
def load_model(model_path):
    # Load the saved model
    checkpoint = torch.load(model_path)

    # Load the model from the checkpoint
    model = checkpoint['model_architecture']

    # Load the model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# Define the prediction function
def predict_image(image):
    # Resize the input image to 28x28
    image = Image.fromarray(image).resize((28, 28))
    image = np.array(image)

    # Preprocess the image (convert to grayscale and normalize)
    image = image.mean(axis=2)  # Convert to grayscale
    image = image / 255.0  # Normalize
    image = image.flatten()

    # Preprocess the image (reshape and convert to tensor)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()  # Get the predicted class index

    return str(prediction)

# Load the trained model
model_path = os.path.join('weights.pth')
model = load_model(model_path)

# Create Gradio interface
demo = gr.Interface(fn=predict_image, inputs='image', outputs='label', title="Simple MNIST classifier", description="Upload an image or draw on the canvas to classify.")
demo.launch(server_port=8080, server_name="0.0.0.0")
