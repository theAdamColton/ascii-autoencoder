import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import bpdb


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super(UnFlatten, self).__init__()
        self.size = size
    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)

class ArgMax(nn.Module):
    def forward(self, input):
        return torch.argmax(input, 1)


class VAE(nn.Module):
    def __init__(self, n_channels=3, h_dim=512, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # Input batchsize x n_channels x 32 x 32
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            # Input batchsize x n_channels * 2 x 16 x 16
            nn.Conv2d(n_channels * 2, n_channels * 3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n_channels * 3),
            nn.ReLU(),
            # Input batchsize x n_channels * 3 x 8 x 8
            nn.Conv2d(n_channels * 3, n_channels * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            # Input batchsize x n_channels * 4 x 4 x 4
            nn.Conv2d(n_channels * 4, h_dim, kernel_size=2, stride=4, padding=0),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(),
            # Input batchsize x h_dim x 1 x 1
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(h_dim),
            nn.ConvTranspose2d(h_dim, n_channels * 4, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n_channels * 3),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 3, n_channels * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=2, stride=2, padding=0),
            # Applies softmax to every channel
            # This only makes sense if using one hot encoding
            nn.Softmax(dim=1),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size(), device=std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar

