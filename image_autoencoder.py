import torch
import torch.utils.tensorboard
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
from font_renderer import ContinuousFontRenderer

from base_vae import BaseVAE


class ImageEncoderVAE(nn.Module):
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


class LitImageAutoencoder(BaseVAE):
    def __init__(self, lr=5e-4, train_dataloader=None, kl_coeff=1.0, save_im_every=10):
        super().__init__()
        self.lr = lr
        self.train_dataloader_obj = train_dataloader
        self.encoder = ImageEncoderVAE()
        self.decoder = ImageDecoder()
        self.random_roll = augmentation.RandomRoll(20, sigma=5)
        self.mse_loss = nn.MSELoss(size_average=True)
        self.kl_coeff = kl_coeff
        self.save_im_every = save_every
        self.font_renderer = ContinuousFontRenderer(
                res = 12, device=torch.device('cuda'), zoom = 21
        )

    def step(self, x, batch_idx):
        """returns loss, logs"""
        # X is an input vector of one hot or close to one hot batch_size x 95 x 64 x 64
        x = self.random_roll(x)
        x = self.font_renderer(x)
        z, x_hat, p, q = self._run_step(x)
        recon_loss = self.mse_loss(x, x_hat)
        kl = torch.distributions.kl_divergence(q, p).mean()
        kl *= self.kl_coeff

        logs = {
            "im_vae_loss": recon_loss,
            "im_vae_kl": kl,
        }

        return loss, logs

