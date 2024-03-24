import os
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the logistic regression model class
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# Define the function to load the model
def load_model(model_path):
    # Load the saved model
    checkpoint = torch.load(model_path)

    # Reinitialize the model with the saved input size and number of classes
    model = LogisticRegression(checkpoint['input_size'], checkpoint['num_classes'])

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

    # Preprocess the image (reshape and convert to tensor)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.view(-1, 28*28))
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()  # Get the predicted class index

    return str(prediction)
# Load the trained model
model_path = os.path.join('experiments', 'models', 'logistic_regression_model.pth')  # Adjust the path as needed
model = load_model(model_path)

# # Create Gradio interface
# image = gr.inputs.Image(shape=(28, 28), image_mode='L', source="canvas")  # image_mode='L' for grayscale
# label = gr.outputs.Label(num_top_classes=1)

demo = gr.Interface(fn=predict_image, inputs='image', outputs='label', title="Simple Image Classifier", description="Upload an image or draw on the canvas to classify.")
demo.launch()
