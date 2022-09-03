import torch
import torch.nn as nn


class Conv2dDownscale(nn.Module):
    """
    Halves the input res
    """

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
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers.forward(x)


class BilinearConvUpsample(nn.Module):
    """
    Multiplies the resolution by 2
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, scale: float = 2.0):
        super().__init__()

        # This sets the zero_pad so that the conv2d layer will have
        # the same output width and height as its input
        assert kernel_size % 2 == 1
        zero_pad = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear"),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=zero_pad
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers.forward(x)


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
