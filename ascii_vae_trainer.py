import torch
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax
import pytorch_lightning as pl
import bpdb

import sys
from os import path

from autoencoder_models import VariationalEncoder, Decoder
from edge_detector import EdgeDetector
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

from base_vae import BaseVAE


class LightningOneHotVAE(BaseVAE):
    """
    VariationalAutoEncoder LightningModule for one hot encoding along n_channels.
    """

    def __init__(
        self,
        font_renderer,
        train_dataloader,
        val_dataloader=None,
        lr=5e-5,
        print_every=10,
        char_weights=None,
        ce_recon_loss_scale=1.0,
        image_recon_loss_coeff=1.0,
        kl_coeff=1.0,
        gumbel_tau=0.9,
        device=torch.device('cuda'),
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
        self.mse_loss = torch.nn.MSELoss(size_average=True)
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
        self.discrete_font_renderer = FontRenderer(
            res=self.font_renderer.font_res,
            zoom=self.font_renderer.zoom,
            device=device,
        )
        self.edge_detector = EdgeDetector(device=device, res=15, sigma=0.2)

    def calculate_image_loss(self, x_hat, x):
        base_image = self.font_renderer.render(x)
        recon_image = self.font_renderer.render(x_hat)

        # This step extentuates the edges on both images. This is intended to
        # reproduce the gestalt effect that humans experience when looking at a
        # segmented edge with a common shape.

        base_image_e = self.edge_detector(base_image)
        recon_image_e = self.edge_detector(recon_image)

       # import vis
       # vis.side_by_side(base_image_e[0][0], recon_image_e[0][0])
       # bpdb.set_trace()

        recon_loss = self.mse_loss(base_image_e, recon_image_e)
        recon_loss *= self.image_recon_loss_coeff
        return recon_loss

    def calculate_ce_loss(self, x_hat, x):
        ce_recon_loss = self.ce_loss(x_hat, x.argmax(dim=1))
        ce_recon_loss *= self.ce_recon_loss_scale
        return ce_recon_loss

    def step(self, x, batch_idx):
        """Returns loss, logs"""
        # Will augment the input batch
        x = self.random_roll(x)
        z, x_hat, p, q = self._run_step(x)

        # CE Loss between original categorical vectors and reconstructed vectors
        if self.ce_recon_loss_scale > 0.0:
            ce_recon_loss = self.calculate_ce_loss(x_hat, x)
        else:
            ce_recon_loss = 0.0
        # Image reconstruction loss
        if self.image_recon_loss_coeff > 0.0:
            # Gumbel step, with a non discrete forward, and backwards
            x_hat_gumbel = gumbel_softmax(x_hat, dim=1, tau=self.gumbel_tau)
            image_recon_loss = self.calculate_image_loss(x_hat_gumbel, x)
        else:
            image_recon_loss = 0.0

        recon_loss = image_recon_loss + ce_recon_loss

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss
        logs = {
            "im_loss": image_recon_loss,
            "ce_loss": ce_recon_loss,
            "kl_loss": kl,
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
                x_recon_gumbel = gumbel_softmax(x_recon, dim=1, tau=self.gumbel_tau)

                # Renders images
                base_image = self.font_renderer.render(x.unsqueeze(0))
                recon_image = self.font_renderer.render(x_recon_gumbel)
                base_image_e = self.edge_detector(base_image)
                recon_image_e = self.edge_detector(recon_image)
                side_by_side = torch.concat((base_image_e, recon_image_e), dim=2).squeeze(0)
                # Logs images
                self.logger.experiment.add_image(
                    "epoch {}".format(self.current_epoch), side_by_side, 0
                )

                x_str = ascii_util.one_hot_embedded_matrix_to_string(x)
                x_recon_str = ascii_util.one_hot_embedded_matrix_to_string(
                    x_recon_gumbel.squeeze(0)
                )
                side_by_side = ascii_util.horizontal_concat(x_str, x_recon_str)
                print(side_by_side)
                print(label)

            self.train()
