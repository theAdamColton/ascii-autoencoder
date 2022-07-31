import torch
import torch.nn.functional as F
import torch.nn as nn
import bpdb

from generic_nn_modules import Flatten, GenericUnflatten


class Decoder(nn.Module):
    """Generic decoder, with a single linear input layer, multiple
    ConvTranspose2d upscaling layers, and batch normalization. Works on a 64x64
    image output size,
    Outputs are scaled by the sigmoid function to be between zero and one.
    """

    def __init__(self, n_channels, z_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
            nn.Linear(z_dim, z_dim * 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim * 4),
            GenericUnflatten((z_dim, 2, 2)),
            nn.ConvTranspose2d(
                z_dim, n_channels * 8, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(
                n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.ConvTranspose2d(
                n_channels * 8, n_channels * 4, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.ConvTranspose2d(
                n_channels * 4, n_channels * 2, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.ConvTranspose2d(
                n_channels * 2, n_channels, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.decoder(z)
        return out


class VariationalEncoder(nn.Module):
    """
    Variational encoder, with a single output linear layer, and then another
    single linear layer producing either mu or sigma^2. This encoder works only
    with 64x64 input image resolution and a 128 latent dimension.
    """

    def __init__(self, n_channels, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input batchsize x n_channels x 64 x 64
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 2),
            nn.Conv2d(
                n_channels * 2, n_channels * 4, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 4),
            nn.Conv2d(
                n_channels * 4, n_channels * 8, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(
                n_channels * 8, n_channels * 8, kernel_size=2, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels * 8),
            nn.Conv2d(n_channels * 8, z_dim, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(z_dim),
            Flatten(),
            nn.Linear(z_dim * 4, z_dim),
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
