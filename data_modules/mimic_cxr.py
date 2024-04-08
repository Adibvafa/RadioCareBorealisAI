import os
import logging
import math
from typing import Optional, Callable
import json
import torch
import pydicom
import pandas as pd
import torch.nn as nn
import torchaudio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale, Resize, CenterCrop, ToTensor
from torchvision import transforms
from typing import List, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer




class MimicIVCXR(Dataset):
    """A PyTorch Dataset for loading image-text pairs from MIMIC-IV-CXR dataset.
    Parameters
    ----------
    data_root: str
        Path to the csv file containing all paths of the image dataset.
    tokenizer - tokenize text
    max_length - maximum length of the text input
    transform: callable
        Torch transform applied to images.
    """

    def __init__(self,
                 data_root: str,
                 tokenizer: str,
                 max_length: int,
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                ) -> None:
        
        """Initialize the dataset."""
        self.transform = transform or Compose([Resize(224),
                                               CenterCrop(224),
                                               ToTensor()])
        self.tokenizer = eval(tokenizer)
        self.max_length = max_length

        # load the csv file
        df = pd.read_csv(data_root)
        
        df = df[:43]
        df = df[df["gender"]!= ""]

        # extracts all the radiograph path from the dataset
        self.images_paths = df["radiograph_path"].tolist()

        # extracts all the radiology reports path from the dataset
        self.text_paths = df["radio_report_path"].tolist()

        # extracts all the labels (gender) from the dataset
        self.labels = df["gender"].tolist()


    def __getitem__(self, idx: int) -> Union[int, torch.Tensor, str,str]:
        """Return the image at the specified index."""

        image_path = "data/" + self.images_paths[idx]
        text_path = "data/" + self.text_paths[idx]

        # Load image
        # Load the JPEG image
        jpeg_image = Image.open(image_path)

        # Convert the image to RGB format
        rgb_image = jpeg_image.convert('RGB')
        image = self.transform(rgb_image)

        # Load text
        with open(text_path, 'r') as file:
            text = file.read()


        if self.tokenizer:
            tokenized_text = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokenized_text['input_ids']
            attention_mask = tokenized_text['attention_mask']
            text = {'input_ids': input_ids, 'attention_mask': attention_mask}


        label = torch.tensor(1 if self.labels[idx] == "Male" else 0, dtype=torch.long)  # Example conversion assuming "Male" is encoded as 1, "Female" as 0

        # return the index, tensor (image), the radiology report and the label (gender)
        return [image, text, label]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images_paths)


def main() -> None:
    data_root = "data/graph_report.csv"
    dataset = MimicIVCXR(data_root,None)
    print(dataset.__getitem__(1772))

if __name__ == "__main__":
    main()
