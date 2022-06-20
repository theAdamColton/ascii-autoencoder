import torch
import pytorch_lightning as pl
import bpdb

from autoencoder_models import VariationalAutoEncoder, VAELoss
import ascii_util

class LightningVAE(pl.LightningModule):
    """
    Lightning VAE Trainer

        only works for ascii art represented by one hot encodings
    """
    def __init__(self, autoencoder: VariationalAutoEncoder, loss: VAELoss, lr: float, print_every=10):
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.loss = loss
        self.lr = lr
        self.print_every = print_every

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_recon, mu, log_var = self.autoencoder(x)
        loss = self.loss(x, mu, log_var, x_recon)
        
        #self.log("Training loss: ", loss)

        if batch_idx % self.print_every == 0:
            with torch.no_grad():
                self.autoencoder.eval()

                # Gets the first item of the batch
                x = x[0]
                x_recon = x_recon[0]
                label = label[0]

                x_str = ascii_util.one_hot_embedded_matrix_to_string(x)
                x_recon_str = ascii_util.one_hot_embedded_matrix_to_string(x_recon)
                side_by_side = ascii_util.horizontal_concat(x_str, x_recon_str)
                print(side_by_side)
                print(label)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

