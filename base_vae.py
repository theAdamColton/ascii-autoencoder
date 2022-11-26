import torch
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import pytorch_lightning as pl

import bpdb

import sys
from os import path

from autoencoder_models import VariationalEncoder, Decoder
from ssim import SSIM
import vis

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util

sys.path.insert(0, path.join(dirname, "./ascii-art-augmentation/"))
import augmentation

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import FontRenderer


class BaseVAE(pl.LightningModule):
    """
    Generic VariationalAutoEncoder LightningModule

    Must inherit step(), self.lr, self.encoder, self.decoder
    """

    def train_dataloader(self):
        return self.train_dataloader_obj

    def val_dataloader(self):
        return self.val_dataloader_obj

    def forward(self, x):
        """Returns recon_x, mu, log_var"""
        mu, log_var = self.encoder(x)
        p, q, z = self.sample(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def _run_step(self, x):
        """Returns z, recon_x, p, q"""
        mu, log_var = self.encoder(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        """The reparameterization trick
        returns p, q, z
        """
        std = torch.exp(log_var / 2)
        std = std.clamp(1e-7)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def calculate_image_loss(self, x_hat, x):
        base_image = self.font_renderer.render(x)
        recon_image = self.font_renderer.render(x_hat)
        image_recon_loss = self.mse_loss(
            base_image.unsqueeze(1), recon_image.unsqueeze(1)
        )
        image_recon_loss *= self.image_recon_loss_coeff
        return image_recon_loss

    def calculate_ce_loss(self, x_hat, x):
        ce_recon_loss = self.ce_loss(x_hat, x.argmax(dim=1))
        ce_recon_loss *= self.ce_recon_loss_scale
        return ce_recon_loss

    def step(self, x, batch_idx):
        raise NotImplementedError("Inheriting class must implement")

    def training_step(self, batch, batch_idx):
        """Returns loss"""
        x, label = batch
        loss, logs = self.step(x, batch_idx)

        self.log_dict(
            {f"t_{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        loss, logs = self.step(x, batch_idx)
        self.log_dict(
            {f"v_{k}": v for k, v in logs.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def init_weights(self, std=0.15):
        """If std is set above ~0.3, there are overflow errors on the first iteration"""
        for _, param in self.named_parameters():
            param.data.normal_(mean=0.0, std=std)
