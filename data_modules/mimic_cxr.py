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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ViTImageProcessor


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
                 graph_report_dir: str,
                 tokenizer: str,
                 max_length: int,
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                ) -> None:
        
        """Initialize the dataset."""
        self.data_root = data_root
        self.transform = transform or Compose([Resize(224),
                                               CenterCrop(224),
                                               ToTensor()])
        self.tokenizer = eval(tokenizer)
        self.max_length = max_length

        # load the csv file
        df = pd.read_csv(graph_report_dir)
        
        df = df[df["gender"]!= ""]

        # extracts all the radiograph path from the dataset
        self.images_paths = df["radiograph_path"].tolist()

        # extracts all the radiology reports path from the dataset
        self.text_paths = df["radio_report_path"].tolist()

        # extracts all the labels (gender) from the dataset
        self.labels = df["gender"].tolist()


    def __getitem__(self, idx: int) -> Union[int, torch.Tensor, str,str]:
        """Return the image at the specified index."""

        image_path = self.data_root + self.images_paths[idx]
        text_path = None # self.data_root + self.text_paths[idx]

        # Load image
        # Load the JPEG image
        jpeg_image = Image.open(image_path)

        # Convert the image to RGB format
        rgb_image = jpeg_image.convert('RGB')

        # Preprocess the image
        image = self.transform(images=rgb_image, return_tensors="pt")['pixel_values'].squeeze()

        # Load text
        # with open(text_path, 'r') as file:
        #     text = file.read()

        # if self.tokenizer:
        #     tokenized_text = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        #     input_ids = tokenized_text['input_ids']
        #     attention_mask = tokenized_text['attention_mask']
        #     text = {'input_ids': input_ids, 'attention_mask': attention_mask}

        label = torch.tensor(1 if self.labels[idx] == "Male"
                             else 0, dtype=torch.long)  # Example conversion assuming "Male" is encoded as 1, "Female" as 0

        # return the index, tensor (image), the radiology report and the label (gender)
        return [image, label, label]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images_paths)


def main() -> None:
    DATA_DIR  = 'mimic-data/'
    ROOT = os.path.dirname(os.getcwd())
    os.chdir(f'E:/RadioCareBorealisAI')

    dataset = MimicIVCXR(data_root=DATA_DIR, 
    graph_report_dir=f"{DATA_DIR}/graph_report.csv", 
    tokenizer="AutoTokenizer.from_pretrained('bert-base-uncased')", 
    max_length=3000, 
    transform=ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    )
    print(dataset.__getitem__(20000))

if __name__ == "__main__":
    main()
