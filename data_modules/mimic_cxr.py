"""
mimic_cxr.py

A PyTorch Dataset for loading image-text pairs from the MIMIC-IV-CXR dataset.
This dataset is used for training and evaluating models that combine image and text data,
specifically radiographs and corresponding radiology reports, along with gender labels.

Classes:
---------
- MimicIVCXR:
  - Initializes the dataset with image paths, text paths, and gender labels.
  - Loads and transforms images.
  - Tokenizes text if a tokenizer is provided.
  - Provides image, text, and label tensors for a given index.
  - Returns the length of the dataset.

Functions:
----------
- main():
  - Demonstrates the usage of the MimicIVCXR dataset.
"""

from typing import Optional, Callable, Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

import pandas as pd
from PIL import Image


class MimicIVCXR(Dataset):
    """
    A PyTorch Dataset for loading image-text pairs from MIMIC-IV-CXR dataset.

    Parameters
    ----------
    data_root : str
        Path to the CSV file containing all paths of the image dataset.
    max_length : int
        Maximum length of the text input.
    tokenizer : Optional[Callable], optional
        Tokenize text, by default None.
    transform : Optional[Callable[[Image.Image], torch.Tensor]], optional
        Torch transform applied to images, by default applies Resize, CenterCrop, and ToTensor.
    """

    def __init__(self,
                 data_root: str,
                 max_length: int,
                 tokenizer: Optional[Callable] = None,
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None) -> None:
        """Initialize the dataset."""
        self.transform = transform or Compose([Resize(224), CenterCrop(224), ToTensor()])
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the CSV file
        df = pd.read_csv(data_root)

        # Remove all rows where gender is null
        df = df[df["gender"] != ""]

        # Extract paths and labels from the dataset
        self.images_paths = df["radiograph_path"].tolist()
        self.text_paths = df["radio_report_path"].tolist()
        self.labels = df["gender"].tolist()

    def __getitem__(self, idx: int) -> Union[torch.Tensor, dict, torch.Tensor]:
        """
        Return the image and text at the specified index along with the label.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        image : torch.Tensor
            The transformed image tensor.
        text : Union[str, dict]
            The radiology report or tokenized text.
        label : torch.Tensor
            The label tensor (1 for Male, 0 for Female).
        """
        image_path = "data/" + self.images_paths[idx]
        text_path = "data/" + self.text_paths[idx]

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Load text
        with open(text_path, 'r') as file:
            text = file.read()

        if self.tokenizer:
            tokenized_text = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = {'input_ids': tokenized_text['input_ids'], 'attention_mask': tokenized_text['attention_mask']}

        # Convert label to tensor
        label = torch.tensor(1 if self.labels[idx] == "Male" else 0, dtype=torch.long)

        return image, text, label

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Number of items in the dataset.
        """
        return len(self.images_paths)


def main() -> None:
    """A function to demonstrate the usage of the MimicIVCXR dataset."""
    data_root = "data/graph_report.csv"
    dataset = MimicIVCXR(data_root=data_root, max_length=512)
    print(dataset.__getitem__(1772))


if __name__ == "__main__":
    main()
