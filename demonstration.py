"""
Traverses the latent space
Make sure that your terminal is large enough, or curses will throw an error
"""
import random
import torch
import numpy as np
import argparse
from os import path
import curses
import time

from dataset import AsciiArtDataset
from autoencoder_models import VanillaAutoenc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=float, default=10)
    parser.add_argument("--steps", dest="steps", type=float, default=100)
    parser.add_argument("--hold-length", dest="hold_length", default=0.5, type=float)
    parser.add_argument(
        "--smooth-factor",
        dest="smooth_factor",
        default=0.5,
        type=float,
        help="Any number in [0,1], represents the smoothing between the different embeddings",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=path.join(
            path.dirname(__file__),
            "models/autoenc_vanilla_deep_cnn_one_hot_64_with_noise",
        ),
    )

    return parser.parse_args()


def main(stdscr, args):
    dataset = AsciiArtDataset(
        res=64, embedding_kind="one-hot", load_autoenc_embeddings=True
    )
    z_dim = 256

    autoenc = VanillaAutoenc(n_channels=95, z_dim=z_dim)
    autoenc.load_state_dict(
        torch.load(path.join(args.model_dir, "autoencoder.pth.tar"))
    )
    autoenc.eval()

    cuda = torch.cuda.is_available()
    if cuda:
        autoenc.cuda()
        Tensor = torch.cuda.FloatTensor
        device = torch.device("cuda")
    else:
        Tensor = torch.FloatTensor
        device = torch.device("cpu")

    curses.noecho()
    curses.curs_set(False)
    pad = curses.newpad(80, 80)

    next_frame = time.time() + 1 / args.frame_rate
    while True:
        embedding1, embedding2 = get_random(device, dataset)

        for x in np.linspace(0, 1, 100):

            if time.time() > next_frame:
                next_frame = time.time() + 1 / args.frame_rate

            x_scaled = np.log10(x ** args.smooth_factor + 1) * 3.322
            with torch.no_grad():
                interp_embedding = x_scaled * embedding1 + (1 - x) * embedding2
                decoded = autoenc.decoder(interp_embedding.unsqueeze(0))
                decoded_str = dataset.decode(decoded[0])

            pad.addstr(0, 0, decoded_str)
            pad.refresh(0, 0, 0, 0, 64, 64)

            if x==0:
                time.sleep(args.hold_length)

        time.sleep(args.hold_length)


def get_random(device, dataset):
    embedding1 = dataset[random.randint(0, len(dataset) - 1)][1]
    embedding1 = embedding1.to(device)
    embedding2 = dataset[random.randint(0, len(dataset) - 1)][1]
    embedding2 = embedding2.to(device)
    return embedding1, embedding2


if __name__ in {"__main__", "__console__"}:
    args = get_args()
    curses.wrapper(main, args)
