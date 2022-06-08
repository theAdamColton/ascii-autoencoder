import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from generic_nn_modules import Flatten, UnFlatten, GenericUnflatten, ArgMax 


class Decoder(nn.Module):
    """Generic decoder, with a single linear input layer, multiple
    ConvTranspose2d upscaling layers, and batch normalization. Works on a 64x64
    image output size,
    Outputs are scaled by the sigmoid function to be between zero and one.
    """
    def __init__(self, n_channels, z_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim * 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim * 4),
            GenericUnflatten((z_dim, 2, 2)),
            nn.ConvTranspose2d(z_dim, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    """
    Variational encoder, with a single output linear layer, and then another
    single linear layer producing either mu or sigma^2. This encoder works only
    with 64x64 input image resolution.
    """
    def __init__(self, n_channels, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
        # Input batchsize x n_channels x 64 x 64
        nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels),
        nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding='same'),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels),
        nn.Conv2d(n_channels, n_channels * 2, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels * 2),
        nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels * 4),
        nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels * 8),
        nn.Conv2d(n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(),
        nn.BatchNorm2d(n_channels * 8),
        nn.Conv2d(n_channels * 8, z_dim, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(),
        nn.BatchNorm2d(z_dim),
        Flatten(),
        nn.Linear(z_dim*4, z_dim),
        nn.LeakyReLU(),
        nn.BatchNorm1d(z_dim),
        )

        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.var_layer = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        """Returns mu, var"""

        p_x = self.encoder(x)
        mu = self.mu_layer(p_x)
        var = self.var_layer(p_x)

        return mu, var


class VAELoss(nn.Module):
    """From: https://github.com/geyang/variational_autoencoder_pytorch"""
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False

    # question: how is the loss function using the mu and variance?
    def forward(self, x, mu, log_var, recon_x):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size


class VariationalAutoEncoder(nn.Module):
    """From: https://github.com/geyang/variational_autoencoder_pytorch"""
    def __init__(self, n_channels, z_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder(n_channels, z_dim)
        self.decoder = Decoder(n_channels, z_dim)

    def forward(self, x):
        """Returns recon_x, mu, log_var"""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space.
        In order for the back-propagation to work, we need to be able to calculate the gradient. 
        This reparameterization trick first generates a normal distribution, then shapes the distribution
        with the mu and variance from the encoder.
        
        This way, we can can calculate the gradient parameterized by this particular random instance.
        """
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
