import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import bpdb


class Flatten(nn.Module):
    def forward(self, input):
        return input.flatten(start_dim=1, end_dim=-1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class GenericUnflatten(nn.Module):
    def __init__(self, shape):
        super(GenericUnflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class ArgMax(nn.Module):
    def forward(self, input):
        return torch.argmax(input, 1)


class VanillaAutoenc(nn.Module):
    """For 64x64 images"""
    def __init__(self, n_channels=3, z_dim=512, categorical=True):
        super(VanillaAutoenc, self).__init__()
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
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
        )

        decoder_layers = [
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim),
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
            nn.LeakyReLU(),
            nn.BatchNorm2d(n_channels),
        ]

        if categorical:
            # Applies softmax to every channel
            # This only makes sense if using one hot encoding
            decoder_layers.append(nn.Softmax(dim=1))
        else:
            decoder_layers.append(nn.Sigmoid)

        self.decoder = nn.Sequential(
            *decoder_layers
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)
