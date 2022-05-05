"""
Train an autoencoder on the ascii dataset
"""


import numpy as np
import torch
import argparse
import torch.nn as nn
import os.path as path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn


from dataset import AsciiArtDataset


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="test_run",
    )
    parser.add_argument(
        "-s",
        "--save-every",
        dest="save_every",
        type=int,
        help="save every n epochs",
        default=100,
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        dest="n_epochs",
        default=200,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument("-c", "--channels", dest="channels", type=int, default=8)
    parser.add_argument("-r", "--res", dest="res", type=int, default=36)
    parser.add_argument("-p", "--preview_every", type=int, default=0)
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")
    parser.add_argument("--noise-std", dest="noise_std", type=float, default=0.022)
    parser.add_argument("--noise-mean", dest="noise_mean", type=float, default=0)
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="autoenc_run",
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument("-lr", dest="learning_rate", default=5e-5, type=float)
    args = parser.parse_args()

    char_embed_path = path.join(
        path.dirname(__file__), "character_embeddings/character_embeddings{}d.npy".format(args.channels)
    )

    dataset = AsciiArtDataset(
        res=args.res,
        added_noise_std=args.noise_std,
        added_noise_mean=args.noise_mean,
        should_add_noise=True,
        character_embeddings=char_embed_path,
    )
    tdataset = dataset.to_tensordataset(torch.device("cuda"))
    dataloader = DataLoader(
        tdataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=True,
    )

def save(path: str, autoenc: Autoencoder, epochs: int):
    torch.save(autoenc, "{}/{}epochs.pt".format(path, epochs))
    with open(os.path.join(models_dir, "../", "epoch"), "w") as f:
        f.write(str(epoch))


class Autoencoder(nn.Module):
    def __init__(self, n_channels, imdim, latent_dim):
        """Square autoencoder"""
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=5),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=5),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ in {"__main__", "__console__"}:
    main()
