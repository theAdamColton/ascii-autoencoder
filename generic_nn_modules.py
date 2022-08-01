import torch
import torch.nn as nn


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
    def __init__(self, *shape):
        super(GenericUnflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class ArgMax(nn.Module):
    def forward(self, input):
        return torch.argmax(input, 1)
