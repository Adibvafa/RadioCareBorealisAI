"""
cnn_model.py

This script defines a convolutional neural network (CNN) model that combines image and text data for classification tasks. The model is trained, validated, and evaluated using data from the Mimic-IV CXR dataset.

Functions:
- train: Trains the CNN model.
- validate: Validates the CNN model.
- evaluate: Evaluates the CNN model on test data.
- main: Main function to load data, create the model, and perform training, validation, and evaluation.

Classes:
- config: Configuration class for model parameters.
- CNNModel: Defines the CNN model architecture.
"""

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from data_modules.mimic_cxr import MimicIVCXR

class config:
    seed = 23
    train_batch_size = 10
    valid_batch_size = 10
    test_batch_size = 10
    learning_rate = 0.001
    num_epochs = 2
    num_classes = 2


class CNNModel(nn.Module):
    """
    CNNModel defines the convolutional neural network architecture that combines image and text data.
    
    Args:
        input_shape (tuple): Shape of the input images.
        text_input_size (int): Size of the text input feature vector.
        in_channels (int, optional): Number of input channels for the image. Defaults to 3.
        output_size (int, optional): Size of the output layer. Defaults to 1.
    """
    def __init__(self, input_shape: tuple, text_input_size: int, in_channels: int = 3, output_size: int = 1):
        super(CNNModel, self).__init__()
        # Define convolutional layers for image processing
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # Define fully connected layers for text processing
        self.fc_text = nn.Linear(text_input_size, 512)

        # Define final fully connected layer
        self.fc_final = nn.Linear(13056, output_size)

        # Define dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Adaptive average pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, image: torch.Tensor, text: dict) -> torch.Tensor:
        """
        Forward pass for the CNN model.
        
        Args:
            image (torch.Tensor): Input image tensor.
            text (dict): Dictionary containing text input IDs.
        
        Returns:
            torch.Tensor: Output tensor from the model.
        """
        # Process image data
        x_image = self.conv1(image)
        x_image = self.relu(x_image)
        x_image = self.maxpool(x_image)

        x_image = self.conv2(x_image)
        x_image = self.relu(x_image)
        x_image = self.maxpool(x_image)

        x_image = self.conv3(x_image)
        x_image = self.relu(x_image)
        x_image = self.maxpool(x_image)

        x_image = self.conv4(x_image)
        x_image = self.relu(x_image)
        x_image = self.maxpool(x_image)

        # Adaptive average pooling
        x_image = self.adaptive_pool(x_image)

        # Flatten the spatial dimensions
        x_image = torch.flatten(x_image, start_dim=1)

        # Process text data
        x_text = self.fc_text(text['input_ids'].float())
        x_text = self.relu(x_text)
        x_text = self.dropout(x_text)

        # Correctly expand x_text to match the batch size of x_image
        x_text = x_text.squeeze(1)

        # Concatenate image and text features
        x_combined = torch.cat((x_image, x_text), dim=1)

        # Final fully connected layer
        x_final = self.fc_final(x_combined)

        return x_final


def train(model: nn.Module, train_dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> float:
    """
    Trains the CNN model.
    
    Args:
        model (nn.Module): The CNN model to be trained.
        train_dataloader (DataLoader): DataLoader for the training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        float: Average training loss.
    """
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        image, text, label = batch[0], batch[1], batch[2]

        image = image.to(device)
        text['input_ids'] = text['input_ids'].to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    return train_loss
    
def validate(model: nn.Module, val_dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Validates the CNN model.
    
    Args:
        model (nn.Module): The CNN model to be validated.
        val_dataloader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            image, text, label = batch[0], batch[1], batch[2]

            image = image.to(device)
            text['input_ids'] = text['input_ids'].to(device)
            label = label.to(device)

            output = model(image, text)
            loss = criterion(output, label)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    return val_loss

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluates the CNN model on test data.
    
    Args:
        model (nn.Module): The CNN model to be evaluated.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        float: Accuracy of the model on the test data.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            image, text, label = batch[0], batch[1], batch[2]

            image = image.to(device)
            text['input_ids'] = text['input_ids'].to(device)
            label = label.to(device)

            outputs = model(image, text)
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    return accuracy


def main():
    """
    Main function to load data, create the model, and perform training, validation, and evaluation.
    """
    # Set your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "graph_report.csv"

    # Load dataset
    dataset = MimicIVCXR(data_root, "AutoTokenizer.from_pretrained('bert-base-uncased')", 3000)

    # Split dataset into training, validation, and test sets
    train_data, val_test_data = train_test_split(dataset, test_size=0.2)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5)

    # Create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config.valid_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False)

    # Create an instance of the model
    model = CNNModel(input_shape=(3, 224, 224), text_input_size=3000, output_size=config.num_classes)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training and validation loop
    for epoch in range(config.num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate(model, val_dataloader, criterion, device)

        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
