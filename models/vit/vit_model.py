"""
vit_model.py

This script trains and evaluates a Vision Transformer (ViT) model on the MIMIC-CXR dataset.
It includes functions for training the model, evaluating it on a test set, and the main
routine for loading data, setting up the model, and running the training and evaluation.

Classes:
---------
- Config:
  - Defines configuration parameters for training and evaluation.

Functions:
----------
- train_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, 
              train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> tuple:
  - Trains the model and validates it on the validation set.

- evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
  - Evaluates the model on the test set.

- main() -> None:
  - Loads data, sets up the model, and runs training and evaluation.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTConfig, AutoTokenizer
from data_modules.mimic_cxr import MimicIVCXR


class Config:
    """Configuration parameters for training and evaluation."""
    seed = 44
    train_batch_size = 10
    valid_batch_size = 10
    test_batch_size = 10
    num_labels = 2
    num_epochs = 2


def train_model(model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, 
                train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> tuple:
    """
    Train the model and validate it on the validation set.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device to use for computation.

    Returns:
        tuple: The trained model, list of training losses, and list of validation accuracies.
    """
    train_losses = []
    val_accuracies = []

    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0
        for images, text, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, text, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{Config.num_epochs}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return model, train_losses, val_accuracies


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test data.
        device (torch.device): The device to use for computation.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, text, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")


def main() -> None:
    """
    Main routine for loading data, setting up the model, and running training and evaluation.
    """
    # Set your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the data
    data_root = "graph_report.csv"

    # Initialize dataset
    dataset = MimicIVCXR(data_root, AutoTokenizer.from_pretrained('bert-base-uncased'), 3000)

    # Split the dataset into training, validation, and test sets
    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=Config.seed)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=Config.seed)

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=Config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=Config.valid_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=Config.test_batch_size, shuffle=False)

    # Load the model with pre-trained weights and configuration
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", 
        config=ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    )
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Train the model
    trained_model, train_losses, val_accuracies = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, device)

    # Evaluate the model on the test set
    evaluate_model(trained_model, test_dataloader, device)


if __name__ == '__main__':
    main()
