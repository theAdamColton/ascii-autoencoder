import torch
import torch.nn as nn
import sys
import os
import bpdb

import pytorch_lightning as pl

from generic_nn_modules import Conv2dDownscale, BilinearConvUpsample, Flatten, GenericUnflatten
from autoenc_trainers import LightningOneHotVAE

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util

sys.path.insert(0, path.join(dirname, "./ascii-art-augmentation/"))
import augmentation

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import FontRenderer

from base_vae import BaseVAE


class ImageEncoderVAE(BaseVAE):
    """Autoencoder for grayscale 640 x 640 images of rendered text."""
    def __init__(self):
        super().__init__()
        res = 12 * 64
        z_dim = 256
        kernel_size = 5
        
        self.encoder = nn.Sequential(
                # Input: batch size x 1 x 768 x 768
                nn.Conv2d(1, 4, kernel_size=kernel_size, stride=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(4),
                # Input: batch size x 4 x 256 x 256
                nn.Conv2d(4, 8, kernel_size=kernel_size, stride=8, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                # Input: batch size x 8 x 32 x 32
                nn.Conv2d(8, 16, kernel_size=kernel_size, stride=8, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                # Input: batch size x 16 x 4 x 4
                Flatten(),
        )
        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.var_layer = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        """Returns mu, var"""
        p_x = self.encoder(x)
        mu = self.mu_layer(p_x)
        var = self.var_layer(p_x)

class ImageDecoder(nn.Module):
    """Decoder for grayscale 12*64 images, to z_dim of 256
    compression ratio 2560"""
    def __init__(self):
        super().__init__()
        res = 12 * 64
        z_dim = 160
        input_channels = 16
        input_size_res = 4
        kernel_size=5

        self.decoder = nn.Sequential(
                # Input: batch size x 256
                GenericUnflatten(input_channels, input_size_res, input_size_res),
                # Input: batch size x 16 x 4 x 4
                BilinearConvUpsample(16, 8, scale=8.0),
                # Input: batch size x 8 x 32 x 32
                BilinearConvUpsample(8, 4, scale=8.0),
                # Input: batch size x 4 x 256 x 256
                BilinearConvUpsample(4, 1, scale=3.0),
                # Input: batch size x 1 x 768 x 768
                nn.Sigmoid(),
        )

        def forward(x):
            return self.decoder(x)


class LitImageAutoencoder():
    def __init__(self, lr=5e-4, train_dataloader=None):
        super().__init__()
        self.lr = lr
        self.train_dataloader_obj = train_dataloader
        self.encoder = ImageEncoderVAE()
        self.decoder = ImageDecoder()

