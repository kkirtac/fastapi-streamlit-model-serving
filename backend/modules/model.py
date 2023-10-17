"""Class definition for Resnet convolutional neural networks. 
18 layered Resnet is suported in this implementation.
"""
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models

class ResnetLightning(pl.LightningModule):
    """ResnetLightning class definition.
    
    Attributes:
        loss_fn: An instance of nn.CrossEntropyLoss. Computes per frame phase classification error.
        network: An instance of Resnet. (See self.create_network())
    """

    def __init__(self, learning_rate: float, out_features=3, **kwargs) -> None:
        """Instantiates the ResnetLightning class.

        Args:
            learning_rate (float): learning rate of the optimization.
            out_features (int): number of output units (num classes).
        """
        super().__init__(**kwargs)

        # call this to save num_resnet_layers, out_features, learning_rate and class_weights to the checkpoint.
        self.save_hyperparameters()
        # Now possible to access out_features from self.hparams

        # base network
        self.resnet = models.resnet18(pretrained=True)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_features=out_features)

        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward pass of the deep network. Overrides pl.LightningModule.forward().

        Args:
            X (torch.tensor): Input tensor.
                X -> (N=batch_size, C=num_channels, H=height, W=width)

        Returns:
            torch.tensor: network outputs.
        """
        return self.resnet(X)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.tensor:
        """Overrides pl.LightningModule.training_step().
        A single step of training optimization.

        Args:
            batch (Tuple): (X,y), i.e., tuple of input and target tensors.
                X -> (N=batch_size, C=num_channels, H=height, W=width)
                y -> (N=batch_size, )
            batch_idx (int): batch index in the current epoch.

        Returns:
            torch.tensor: the loss tensor.
        """
        x, y = batch
        y_hat = self.resnet(x)
        loss = self.loss_fn(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
