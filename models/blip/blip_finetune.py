"""
blip_finetune.py

This script fine-tunes the BLIP model for conditional generation using the MIMIC-IV CXR dataset. 
It utilizes PyTorch Lightning for efficient training and distributed data parallel strategy.

Classes:
---------
- BLIP_MIMIC_CXR:
  - Initializes the BLIP_MIMIC_CXR module.
  - Performs a single training step.
  - Performs a single validation step.
  - Configures the optimizers and learning rate scheduler.

Functions:
----------
- main():
  - Sets up the environment and prepares the dataset for training.
  - Initializes the processor, model, and dataset.
  - Splits the dataset into training and validation sets.
  - Defines the dataloaders for training and validation.
  - Defines the callbacks for model checkpointing and learning rate monitoring.
  - Initializes the Wandb logger.
  - Initializes the PyTorch Lightning trainer.
  - Starts the training process.
"""

import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from transformers import BlipForConditionalGeneration, BlipProcessor

from utils import seed_everything
from models.blip.blip_dataset import BlipMimicIVCXR


# Constants
SEED = 23
DATASET_LEN = int(424189 * 0.9)
NGPUS = 4
BATCH_SIZE = 32
MAX_EPOCHS = 2
MAX_LEN = 512
ACC = 1
DEBUG = False

TRAIN_BATCHES = DATASET_LEN / BATCH_SIZE / NGPUS * MAX_EPOCHS
GRAD_STEPS = TRAIN_BATCHES * ACC
WARMUP = int(0.1 * GRAD_STEPS)
DECAY = int(0.9 * GRAD_STEPS)


class BLIP_MIMIC_CXR(pl.LightningModule):
    def __init__(self, model: BlipForConditionalGeneration, warmup: int, decay: int) -> None:
        """
        Initializes the BLIP_MIMIC_CXR module.

        Args:
            model (BlipForConditionalGeneration): The BLIP model for conditional generation.
            warmup (int): Number of warmup steps for the learning rate scheduler.
            decay (int): Number of decay steps for the learning rate scheduler.
        """
        super().__init__()
        self.model = model
        self.warmup = warmup
        self.decay = decay

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )

        loss = outputs.loss
        current_lr = self.lr_schedulers().get_last_lr()[0]

        self.log_dict(
            {"loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )

        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, Any]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )

        loss = outputs.loss
        current_lr = self.lr_schedulers().get_last_lr()[0]

        self.log_dict(
            {"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )

        return loss

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            Tuple[List, List]: The optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-5,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup
        )
        
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=self.decay
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
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

    # Initialize processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Prepare dataset
    dataset = BlipMimicIVCXR(
        graph_report_dir=graph_report_dir, 
        tokenizer=None,
        max_length=512,
        transform=processor
    )

    # Split dataset
    train_dataset, val_dataset = train_test_split(
        dataset,
        test_size=0.1,
        random_state=SEED
    )

    pl_model = BLIP_MIMIC_CXR(model, warmup=WARMUP, decay=DECAY)

    # Prepare dataloaders
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

    # Define callbacks
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

    # Initialize logger
    wandb_logger = WandbLogger(
        project='blip-mimic-cxr',
        save_dir='wandb',
        entity='adibvafa',
        resume="allow",
    )

    # Initialize trainer
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

    # Start training
    trainer.fit(
        model=pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )
