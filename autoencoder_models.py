import torch
import torch.nn.functional as F
import torch.nn as nn
import bpdb

from generic_nn_modules import Flatten, GenericUnflatten


class BilinearConvUpsample(nn.Module):
    """
    Doubles the resolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        # This sets the zero_pad so that the conv2d layer will have
        # the same output width and height as its input
        assert kernel_size % 2 == 1
        zero_pad = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=zero_pad
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers.forward(x)


class Decoder(nn.Module):
    """Decoder with a single linear input layer, multiple
    BilinearConvUpsample upscaling layers, and batch normalization. Works on a 64x64
    image output size,
    Outputs are scaled by the softmax function to be have a sum of 1.
    """

    def __init__(self, n_channels=95, z_dim=128, kernel_size=5):
        assert z_dim == 128
        super().__init__()
        input_res = int((z_dim // 8) ** 0.5)
        self.decoder = nn.Sequential(
            # Input size comments assume an input z_dim of 128
            # Input: batch_size by 128
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
            GenericUnflatten(8, input_res, input_res),

            # Input: batch_size by 8 by 4 by 4
            BilinearConvUpsample(8, 8, kernel_size=kernel_size),

            # Input: batch_size by 8 by 8 by 8
            BilinearConvUpsample(8, 16, kernel_size=kernel_size),

            # Input: batch_size by 16 by 16 by 16
            BilinearConvUpsample(16, 32, kernel_size=kernel_size),

            # Input: batch_size by 32 by 32 by 32
            BilinearConvUpsample(32, 64, kernel_size=kernel_size),

            # Input: batch_size by 64 by 64 by 64
            nn.Conv2d(64, n_channels, kernel_size, stride=1, padding=kernel_size // 2),

            # Input: batch_size by 95 by 64 by 64
            nn.Softmax(dim=1),
        )

    def forward(self, z):
        out = self.decoder(z)
        return out


class Conv2dDownscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = 5
        stride = 2
        zero_padding = 2

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=zero_padding,
            ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers.forward(x)


class VariationalEncoder(nn.Module):
    """
    Variational encoder, with a single output linear layer, and then another
    single linear layer producing either mu or sigma^2. This encoder works only
    with 64x64 input image resolution and a 128 latent dimension.
    """

    def __init__(self, n_channels=95, z_dim=128):
        assert n_channels == 95
        assert z_dim == 128
        super().__init__()
        self.encoder = nn.Sequential(
            # Size comments are based on an input shape of batch_size by 95 by
            # 64 by 64
            # Input: batchsize x 95 x 64 x 64
            Conv2dDownscale(95, 64),
            # Input: batchsize x 64 x 32 x 32
            Conv2dDownscale(64, 32),
            # Input: batch_size x 32 x 16 x 16
            Conv2dDownscale(32, 16),
            # Input: batch_size x 16 x 8 x 8
            Conv2dDownscale(16, 8),
            # Input: batch_size x 8 x 4 x 4
            Flatten(),
            # Input: batch_size x 128
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
