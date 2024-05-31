"""
blip_dataset.py

This script includes functions to load a checkpoint for the BLIP model, display an image,
and process text reports. The functions are designed to be used for image captioning tasks 
with the BLIP model from Salesforce.

Classes:
    None

Functions:
    - load_checkpoint(path: str, device: torch.device) -> BlipForConditionalGeneration:
        Loads the model checkpoint and prepares the BLIP model for evaluation.
    - view_image(pixel_values: torch.Tensor) -> None:
        Displays an image given its pixel values.
    - process_report(text: str) -> str:
        Processes a text report by adding newlines before specific keywords.
"""

from typing import Callable, Optional, Union

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class BlipMimicIVCXR(Dataset):
    """
    A PyTorch Dataset for loading image-text pairs from the MIMIC-IV-CXR dataset.

    Parameters
    ----------
    graph_report_dir : str
        Path to the CSV file containing all paths of the image dataset.
    tokenizer : str
        Tokenizer to tokenize text.
    max_length : int
        Maximum length of the text input.
    transform : Optional[Callable[[Image.Image], torch.Tensor]]
        Torch transform applied to images.
    """
    
    def __init__(self,
                 graph_report_dir: str,
                 tokenizer: str,
                 max_length: int,
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None) -> None:
        """Initialize the dataset."""
        self.graph_report_dir = graph_report_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the CSV file
        df = pd.read_csv(self.graph_report_dir)
        
        # Filter out rows with missing gender
        df = df[df["gender"] != ""]

        # Extract all the radiograph paths from the dataset
        self.images_paths = df["radiograph_path"].tolist()

        # Extract all the radiology reports paths from the dataset
        self.text_paths = df["radio_report_path"].tolist()

    def __getitem__(self, idx: int) -> dict:
        """
        Return the image-text pair at the specified index.

        Parameters
        ----------
        idx : int
            Index of the data point to retrieve.

        Returns
        -------
        dict
            Dictionary containing the transformed image and tokenized text.
        """
        image_path = self.images_paths[idx]
        text_path = self.text_paths[idx]

        # Load the JPEG image
        rgb_image = Image.open(image_path)

        # Load and process text
        with open(text_path, 'r') as file:
            text = process_text(file.read())

        # Apply the transformations
        encoding = self.transform(
            images=rgb_image, 
            text=text,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        encoding.pop('attention_mask')

        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        
        return encoding

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.images_paths)


def process_text(text: str) -> str:
    """
    Process the text by normalizing whitespace and converting to lowercase.

    Parameters
    ----------
    text : str
        The text to process.

    Returns
    -------
    str
        The processed text.
    """
    # Strip leading and trailing whitespaces
    text = text.strip()
    
    # Replace multiple consecutive spaces with a single space
    text = text.replace('  ', ' ')
    text = text.replace(' \n \n ', ' ')
    text = text.replace('\n ', '\n')
    
    # Replace multiple consecutive newline characters with a single newline
    text = '\n'.join(filter(None, text.split('\n')))
    
    return text.lower()


def slice_to_period_after_indication(s: str) -> str:
    """
    Slice the text to include content up to the first period or comma after 'indication'.

    Parameters
    ----------
    s : str
        The input text.

    Returns
    -------
    str
        The sliced text.
    """
    s = s.lower()

    # Find the starting index of 'indication'
    start_index = s.find('indication')
    if (start_index == -1):
        finding_index = s.find('finding')
        return s[finding_index:].strip()
    
    # Find the index of the first period after 'indication'
    period_index = s.find('.', start_index)
    if period_index != -1:
        return s[:period_index + 1].strip()

    # Find the index of the first comma after 'indication' if no period is found
    comma_index = s.find(',', start_index)
    if comma_index != -1:
        return s[:comma_index + 1].strip()

    return s
