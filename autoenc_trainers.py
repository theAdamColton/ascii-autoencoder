import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import bpdb

from autoencoder_models import VariationalEncoder, Decoder
import ascii_util

class LightningOneHotVAE(pl.LightningModule):
    """
    VariationalAutoEncoder LightningModule for one hot encoding along n_channels.
    """
    def __init__(self, lr= 5E-5, print_every=10):
        super().__init__()
        z_dim = 128
        n_channels = 95

        self.save_hyperparameters()
        self.lr = lr
        self.print_every = print_every
        self.encoder = VariationalEncoder(n_channels, z_dim)
        self.decoder = Decoder(n_channels, z_dim)
        self.kl_coeff = 1.0

    def forward(self, x):
        """Returns recon_x, mu, log_var"""
        mu, log_var = self.encoder(x)
        p, q, z = self.sample(mu, log_var)
        recon_x = self.decoder(z)
        #recon_x_gumbel = nn.functional.gumbel_softmax(log_recon_x, tau=temperature, hard=True, dim=1)
        #recon_x_gumbel = utils.gumbel_softmax(recon_x, temperature, 128, 95, dim=1)
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

    def step(self, x, batch_idx):
        """Returns loss, logs"""
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.cross_entropy(x_hat, x.argmax(dim=1), reduction="mean")

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss
        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        """Returns loss"""
        x, label = batch
        loss, logs = self.step(x, batch_idx)
        
        if batch_idx % self.print_every == 0:
            with torch.no_grad():
                self.eval()

                # Gets the first item of the batch
                x_0 = x[0]
                x_0_unsqueezed = x_0.unsqueeze(0)
                # Reconstructs the first item of the batch
                x_recon, _, _ = self.forward(x_0_unsqueezed)
                label = label[0]

                x_str = ascii_util.one_hot_embedded_matrix_to_string(x[0])
                x_recon_str = ascii_util.one_hot_embedded_matrix_to_string(x_recon[0])
                side_by_side = ascii_util.horizontal_concat(x_str, x_recon_str)
                print(side_by_side)
                print(label)

            self.train()

        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def init_weights(self, std=0.2):
        """If std is set above ~0.3, there are overflow errors on the first iteration"""
        for _, param in self.named_parameters():
            param.data.normal_(mean=0.0, std=std)

