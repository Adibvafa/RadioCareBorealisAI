"""
blip_dataset.py

This script includes functions to load a checkpoint for the BLIP model, display an image,
and process text reports. The functions are designed to be used for image captioning tasks 
with the BLIP model from Salesforce.

Functions:
----------
- load_checkpoint(path: str, device: torch.device) -> BlipForConditionalGeneration:
  - Loads the model checkpoint and prepares the BLIP model for evaluation.

- view_image(pixel_values: torch.Tensor) -> None:
  - Displays an image given its pixel values.

- process_report(text: str) -> str:
  - Processes a text report by adding newlines before specific keywords.
"""

import torch
import matplotlib.pyplot as plt
from transformers import BlipForConditionalGeneration, BlipProcessor


def load_checkpoint(path: str, device: torch.device) -> BlipForConditionalGeneration:
    """
    Load the model checkpoint and prepare the BLIP model for evaluation.
    
    Args:
        path (str): The path to the checkpoint file.
        device (torch.device): The device to load the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        BlipForConditionalGeneration: The loaded and prepared BLIP model.
    """
    # Load the checkpoint
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    
    # Adjust state_dict keys to match the model's expected keys
    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    state_dict = {key.replace("vision_", "vision_model."): value for key, value in state_dict.items()}
    
    # Load the BLIP model and apply the state_dict
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model


def view_image(pixel_values: torch.Tensor) -> None:
    """
    Display an image given its pixel values.
    
    Args:
        pixel_values (torch.Tensor): The pixel values of the image.
    
    Returns:
        None
    """
    # Convert pixel values to numpy array and display the image
    pixel_values = pixel_values.cpu().numpy()
    plt.imshow(pixel_values.squeeze()[0, :, :], cmap='gray')
    plt.show()


def process_report(text: str) -> str:
    """
    Process a text report by adding newlines before specific keywords.
    
    Args:
        text (str): The text report to be processed.
    
    Returns:
        str: The processed text report with newlines before specified keywords.
    """
    keywords = ['final report', 'examination', 'indication', 'history', 'findings', 'impression', 'comparison', 'exp']
    
    # Add '\n' before each keyword
    for keyword in keywords:
        text = text.replace(keyword, f'\n{keyword}')
    
    return text.strip()
