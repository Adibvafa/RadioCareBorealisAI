import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import torch
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification
from sklearn.model_selection import train_test_split
from transformers import ViTForImageClassification, ViTConfig, ViTImageProcessor
from data_modules.mimic_cxr import MimicIVCXR
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class config:
    seed = 44
    train_batch_size = 10
    valid_batch_size = 10
    test_batch_size = 10
    num_labels = 2
    num_epochs = 2


def train_model(model, criterion, optimizer, train_loader, val_loader, device):
    train_losses = []  # List to store training loss for each epoch
    val_accuracies = []  # List to store validation accuracy for each epoch
    
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        for images, text, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images).logits  # Get logits from model outputs
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
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return model, train_losses, val_accuracies

def evaluate_model(model, test_loader,device):
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


def main():
    # Set your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = "graph_report.csv"

    dataset = MimicIVCXR(data_root,"AutoTokenizer.from_pretrained('bert-base-uncased')",3000)

    train_data, val_test_data = train_test_split(dataset, test_size=0.2)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5)


    # Load your data and create dataloaders
    train_dataloader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config.valid_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k"))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    trained_model, train_losses, val_accuracies = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, device)

    evaluate_model(trained_model, test_dataloader,device)


if __name__=='__main__':
    main()
