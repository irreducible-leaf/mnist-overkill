import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SmallMLP
import logging


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Running training script...")

# set paths to data
DATA_TRAIN_PATH = 'data/train'
LABELS_TRAIN_PATH = 'data/labels_20k.csv'
LABELS_TEST_PATH = 'data/labels_test.csv'

# load labels into memory
logging.info("Loading labels...")
df_labels_train = pd.read_csv(LABELS_TRAIN_PATH)

# load data into memory
def load_data(df_labels, dirpath):
    data = []
    for i, row in df_labels.iterrows():
        loadpath = os.path.join(dirpath, f'{i}.npy')
        img = np.load(loadpath)
        data.append(img)
    return np.array(data)

logging.info("Loading data...")
data_train = load_data(df_labels_train, DATA_TRAIN_PATH)

logging.info(f"Loaded {len(data_train)} training examples and {len(df_labels_train)} labels")

# create a train/val split
logging.info("Creating train/val split...")
labels_train_array = df_labels_train['label'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data_train, labels_train_array, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
logging.info("Converting data and labels into Pytorch Tensors...")
X_train_tensor = torch.from_numpy(X_train).float()
X_val_tensor = torch.from_numpy(X_val).float()
y_train_tensor = torch.from_numpy(y_train)
y_val_tensor = torch.from_numpy(y_val)

# Flatten input data
X_train_tensor = X_train_tensor.view(-1, 28 * 28)
X_val_tensor = X_val_tensor.view(-1, 28 * 28)

# Normalize input data
X_train_tensor = X_train_tensor / 255.0
X_val_tensor = X_val_tensor / 255.0

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# Initialize model
input_size = 28 * 28
hidden_size1 = 32
hidden_size2 = 64
num_classes = 10
model = SmallMLP(input_size, hidden_size1, hidden_size2, num_classes)
logging.info(f"Created model\n\n{model}\n\n")

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed

# define a training loop
def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate average training loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions

        # Evaluate the model on validation data
        val_loss, val_acc = evaluate(model, criterion, val_loader)

        # Print training and validation statistics
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_acc:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_acc:.4f}')

def evaluate(model, criterion, data_loader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate evaluation statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy

num_epochs = 10
train(model, criterion, optimizer, train_loader, val_loader, num_epochs)

# Define the file path to save the model
model_path = 'weights/weights.pth'

# Save the entire model
torch.save({
    'model_state_dict': model.state_dict(),  # Save model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
    'input_size': input_size,  # Save input size
    'hidden_size1': hidden_size1,  # Save hidden layer 1 size
    'hidden_size2': hidden_size2,  # Save hidden layer 2 size
    'num_classes': num_classes,  # Save number of classes
    'model_architecture': model  # Save model architecture
}, model_path)


logging.info(f"Model saved successfully at: {model_path}")