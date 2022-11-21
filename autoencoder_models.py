import torch
import torch.nn.functional as F
import torch.nn as nn

from generic_nn_modules import (
    Flatten,
    GenericUnflatten,
    BilinearConvUpsample,
    Conv2dDownscale,
)


class Decoder(nn.Module):
    """Decoder with a single linear input layer, multiple
    BilinearConvUpsample upscaling layers, and batch normalization. Works on a 64x64
    image output size,


    Outputs are scaled by the softmax function to be have a sum of 1,
    Log probabilities are returned.
    """

    def __init__(self, n_channels=95, z_dim=512, kernel_size=5):
        super().__init__()
        assert z_dim == 512
        input_side_res = 8
        input_channels = 8

        self.decoder = nn.Sequential(
            # Input size comments assume an input z_dim of 256
            # Input: batch_size by 256
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.BatchNorm1d(z_dim),
            # Input: batch_size by 256
            # nn.Linear(z_dim, z_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(z_dim),
            GenericUnflatten(input_channels, input_side_res, input_side_res),
            # Input: batch_size by 8 by 8 by 8
            BilinearConvUpsample(8, 16, kernel_size=kernel_size, scale=3 / 2),
            # Input batch_size by 16 by 12 by 12
            BilinearConvUpsample(16, 32, kernel_size=kernel_size, scale=4 / 3),
            # Input: batch_size by 32 by 16 by 16
            BilinearConvUpsample(32, 48, kernel_size=kernel_size),
            # Input: batch_size by 48 by 32 by 32
            BilinearConvUpsample(48, 64, kernel_size=kernel_size),
            # Input: batch_size by 64 by 64 by 64
            nn.Conv2d(64, n_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(95),
            # Input: batch_size by 95 by 64 by 64
            nn.Conv2d(
                n_channels, n_channels, kernel_size, stride=1, padding=kernel_size // 2
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, z):
        # returns the log probability
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    """
    Variational encoder, with a single output linear layer, and then another
    single linear layer producing either mu or sigma^2. This encoder works only
    with 64x64 input image resolution and a 128 latent dimension.
    """

    def __init__(self, z_dim=512, kernel_size=5):
        super().__init__()
        assert z_dim == 512

        self.encoder = nn.Sequential(
            # Size comments are based on an input shape of batch_size by 95 by
            # 64 by 64
            # Input batch_size x 95 x 64 x 64
            nn.Conv2d(95, 64, kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Input: batchsize x 64 x 64 x 64
            Conv2dDownscale(64, 48),
            # Input: batchsize x 48 x 32 x 32
            Conv2dDownscale(48, 32),
            # Input: batch_size x 32 x 16 x 16
            nn.Conv2d(32, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # Input: batch_size x 16 x 12 x 12
            nn.Conv2d(16, 8, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # Input: batch_size x 8 x 8 x 8
            Flatten(),
            # Input: batch_size x 512
            # nn.Linear(z_dim, z_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(z_dim),
        )

        self.mu_layer = nn.Linear(z_dim, z_dim)
        self.var_layer = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        """Returns mu, var"""
        p_x = self.encoder(x)
        mu = self.mu_layer(p_x)
        var = self.var_layer(p_x)

        return mu, var
