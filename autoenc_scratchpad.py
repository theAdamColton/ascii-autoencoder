from autoencoder_models import VAE

import torch.nn as nn
import torch
from torchsummary import summary

vae = VAE(image_channels=95, h_dim=1024, z_dim=64)