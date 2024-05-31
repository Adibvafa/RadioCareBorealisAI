import os
from typing import Callable, List, Optional, Union

import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from transformers import BlipForConditionalGeneration, BlipProcessor

SEED = 23
DATASET_LEN = int(424189 * 0.9)
NGPUS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 2
MAX_LEN = 512
ACC = 1

TRAIN_BATCHES = DATASET_LEN / BATCH_SIZE / NGPUS * MAX_EPOCHS
GRAD_STEPS = TRAIN_BATCHES * ACC

WARMUP = int(0.1 * GRAD_STEPS)
DECAY = int(0.9 * GRAD_STEPS)

DEBUG = False


def process_text(text):
    # Strip leading and trailing whitespaces
    text = text.strip()
    
    # Replace any tab characters with spaces
    text = text.replace('  ', ' ')
    text = text.replace(' \n \n ', ' ')
    text = text.replace('\n ', '\n')
    
    # Replace multiple consecutive newline characters with a single newline
    text = '\n'.join(filter(None, text.split('\n')))
    
    return text.lower()

def slice_to_period_after_indication(s):
    s = s.lower()

    # Find the starting index of 'indication'
    start_index = s.find('indication')
    if start_index == -1:
        finding_index = s.find('finding')
        return s[finding_index:].strip()
    
    # Find the index of the first '.' after 'indication'
    period_index = s.find('.', start_index)

    if period_index != -1:
        return s[:period_index + 1].strip()

    elif period_index == -1:
        comma_index = s.find(',', start_index)
        return s[:comma_index + 1].strip()

    return s

def seed_everything(seed: int) -> None:
    """Seed all components of the model.

    Parameters
    ----------
    seed: int
        Seed value to use

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


class MimicIVCXR(Dataset):
    """A PyTorch Dataset for loading image-text pairs from MIMIC-IV-CXR dataset.
    Parameters
    ----------
    graph_report_dir: str
        Path to the csv file containing all paths of the image dataset.
    tokenizer - tokenize text
    max_length - maximum length of the text input
    transform: callable
        Torch transform applied to images.
    """

    def __init__(self,
                 graph_report_dir: str,
                 tokenizer: str,
                 max_length: int,
                 transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                ) -> None:
        
        """Initialize the dataset."""
        self.graph_report_dir = graph_report_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load the csv file
        df = pd.read_csv(self.graph_report_dir)
        
        df = df[df["gender"]!= ""]

        # extracts all the radiograph path from the dataset
        self.images_paths = df["radiograph_path"].tolist()

        # extracts all the radiology reports path from the dataset
        self.text_paths = df["radio_report_path"].tolist()

    def __getitem__(self, idx: int) -> Union[int, torch.Tensor, str,str]:
        """Return the image at the specified index."""

        image_path = self.images_paths[idx]
        text_path = self.text_paths[idx]

        # Load image
        # Load the JPEG image
        rgb_image = Image.open(image_path)#.convert('RGB')
        
        # Load text
        with open(text_path, 'r') as file:
            text = process_text(file.read())

        encoding  = self.transform(
            images=rgb_image, 
            text=text,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        encoding.pop('attention_mask')

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        
        return encoding

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images_paths)


class BLIP_MIMIC_CXR(pl.LightningModule):
    def __init__(self, model, warmup, decay):
        super().__init__()
        self.model = model
        self.warmup = warmup
        self.decay = decay

    def training_step(self, batch, batch_idx):
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )

        loss = outputs.loss
        (current_lr,) = self.lr_schedulers().get_last_lr()

        self.log_dict(
            dictionary={"loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )

        loss = outputs.loss
        (current_lr,) = self.lr_schedulers().get_last_lr()

        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(), lr=5e-5,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup
        )
        
        linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.01,
            total_iters=self.decay
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, linear_decay],
            milestones=[self.warmup]
        )
    
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    

if __name__ == '__main__':

    # Setup environment
    seed_everything(SEED)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    graph_report_dir = "mimic-data/graph_report_adib.csv"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    dataset = MimicIVCXR(
        graph_report_dir=graph_report_dir, 
        tokenizer=None,
        max_length=512,
        transform=processor
    )
    # train_dataset, val_dataset = train_test_split(
    #     dataset,
    #     test_size=0.1,
    #     random_state=SEED
    # )

    # Unusual split
    split_index = int(len(dataset) * 0.9)
    train_dataset = Subset(dataset, list(range(0, split_index)))
    val_dataset = Subset(dataset, list(range(split_index, len(dataset))))

    pl_model = BLIP_MIMIC_CXR(model, warmup=WARMUP, decay=DECAY)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        persistent_workers=True
    )

    callbacks = [
    ModelCheckpoint(
        dirpath='checkpoints',
        filename='validated_model_{epoch}',
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True
    ),
    LearningRateMonitor(logging_interval="step"),
]

    wandb_logger = WandbLogger(
        project='blip-mimic-cxr',
        save_dir='wandb',
        entity='adibvafa',
        resume="allow",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=NGPUS,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        log_every_n_steps=20,
        accumulate_grad_batches=ACC,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model=pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )