import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EdgeDetector(nn.Module):
    """
    Laplacian of Guassian

    https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
    """

    def __init__(self, res=7, sigma=1.0, device=torch.device("cuda")):
        super().__init__()
        kernel = gen_filter(res, sigma=sigma)
        self.kernel = kernel.to(device)
        self.padding = res // 2

    def forward(self, x):
        """
        x is a tensor of shape: (batchsize by width by height)
        """
        out = F.conv2d(
            x.unsqueeze(1), self.kernel.unsqueeze(0).unsqueeze(0), padding=self.padding
        )
        return out


def gen_filter(res, sigma=1.0):
    assert res % 2 == 1
    kernel = torch.zeros(res, res)
    middle_index = res // 2
    for x in range(-middle_index, middle_index + 1):
        for y in range(-middle_index, middle_index + 1):
            z = lap_of_gaus(x, y, sigma=sigma)
            x_i = x + middle_index
            y_i = y + middle_index
            # print("{}, {}; {}, {}; : {}".format(x_i, x, y_i, y, z))
            kernel[x_i, y_i] = z

    return kernel


def lap_of_gaus(x, y, sigma=1.0):
    """
    Laplacian of a Guassian
    """
    return (
        -1.0
        / (math.pi * sigma**2)
        * (1 - (x**2 + y**2) / (2 * sigma**2))
        * math.exp(-1.0 * (x**2 + y**2) / (2.0 * sigma**2))
    )
