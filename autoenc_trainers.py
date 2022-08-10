import torch
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import pytorch_lightning as pl

import sys
from os import path

from autoencoder_models import VariationalEncoder, Decoder
from ssim import SSIM

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util

sys.path.insert(0, path.join(dirname, "./ascii-art-augmentation/"))
import augmentation

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import FontRenderer


class LightningOneHotVAE(pl.LightningModule):
    """
    VariationalAutoEncoder LightningModule for one hot encoding along n_channels.
    """

    def __init__(
        self,
        font_renderer,
        font_renderer_res,
        train_dataloader,
        val_dataloader=None,
        lr=5e-5,
        print_every=10,
        char_weights=None,
        ce_recon_loss_scale=0.1,
        image_recon_loss_coeff=1.0,
        kl_coeff=1.0,
        gumbel_tau=0.9,
    ):
        super().__init__()

        self.lr = lr
        self.print_every = print_every
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()
        self.kl_coeff = kl_coeff
        self.font_renderer = font_renderer
        self.ce_recon_loss_scale = ce_recon_loss_scale
        self.image_recon_loss_coeff = image_recon_loss_coeff

        self.l1_loss = torch.nn.L1Loss()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=char_weights)
        self.bce_loss = torch.nn.BCELoss()
        self.ssim_loss = SSIM()
        self.train_dataloader_obj = train_dataloader
        self.val_dataloader_obj = val_dataloader
        self.save_hyperparameters(
            ignore=["train_dataloader", "val_dataloader", "font_renderer"]
        )
        self.random_roll = augmentation.RandomRoll(15, sigma=5)
        self.gumbel_tau = gumbel_tau

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
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def calculate_image_loss(self, x_hat, x, batch_size):
        # Gumbel step, with a non discrete forward, and backwards
        x_hat_gumbel = gumbel_softmax(x_hat, dim=1, tau=self.gumbel_tau)
        base_image = self.font_renderer.render(x)
        recon_image = self.font_renderer.render(x_hat_gumbel)
        image_recon_loss = self.l1_loss(
            base_image.unsqueeze(1), recon_image.unsqueeze(1)
        )
        image_recon_loss /= batch_size
        image_recon_loss *= self.image_recon_loss_coeff
        return image_recon_loss

    def calculate_ce_loss(self, x_hat, x, batch_size):
        ce_recon_loss = self.ce_loss(x_hat, x.argmax(dim=1))
        ce_recon_loss /= batch_size
        ce_recon_loss *= self.ce_recon_loss_scale
        return ce_recon_loss

    def step(self, x, batch_idx):
        """Returns loss, logs"""
        # Will augment the input batch
        x = self.random_roll(x)
        batch_size = x.shape[0]

        z, x_hat, p, q = self._run_step(x)

        # CE Loss between original categorical vectors and reconstructed vectors
        if self.ce_loss:
            ce_recon_loss = self.calculate_ce_loss(x_hat, x, batch_size)
        else:
            ce_recon_loss = 0.0
        # Image reconstruction loss
        if self.image_recon_loss_coeff > 0.0:
            image_recon_loss = self.calculate_image_loss(x_hat, x, batch_size)
        else:
            image_recon_loss = 0.0

        recon_loss = image_recon_loss + ce_recon_loss

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl /= batch_size
        kl *= self.kl_coeff

        loss = kl + recon_loss
        logs = {
            "image_loss": image_recon_loss,
            "ce_loss": ce_recon_loss,
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }

        return loss, logs

    def on_train_epoch_end(self):
        if self.current_epoch % self.print_every == 0:
            x, label = self.train_dataloader().dataset.get_random_training_item()
            x = torch.Tensor(x)
            x = x.to(self.device)
            # Will random roll
            x = self.random_roll(x.unsqueeze(0)).squeeze(0)

            with torch.no_grad():
                self.eval()

                # Reconstructs the item
                x_recon, _, _ = self.forward(x.unsqueeze(0))
                x_recon = x_recon.squeeze(0)
                x_str = ascii_util.one_hot_embedded_matrix_to_string(x)
                x_recon_str = ascii_util.one_hot_embedded_matrix_to_string(x_recon)
                side_by_side = ascii_util.horizontal_concat(x_str, x_recon_str)
                print(side_by_side)
                print(label)

            self.train()

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
