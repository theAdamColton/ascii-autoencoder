"""
Traverses the latent space
Make sure that your terminal is large enough, or curses will throw an error
"""
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from os import path
import curses
import time
import sys

import bpdb

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
import ascii_util

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import ContinuousFontRenderer

from ascii_vae_trainer import LightningOneHotVAE

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--frame-rate", dest="frame_rate", type=float, default=10)
    parser.add_argument("--steps", dest="steps", type=int, default=200)
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
    parser.add_argument(
            "--cuda",
            dest="cuda",
            default=False,
    )
    return parser.parse_args()


def main(stdscr, args):
    dataset = AsciiArtDataset(res=64)

    cuda = args.cuda
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )
    font_renderer = ContinuousFontRenderer(
        device=device,
        res=9,
        zoom=21,
    )

    autoenc = LightningOneHotVAE.load_from_checkpoint(
        args.model_dir,
        font_renderer=font_renderer,
        train_dataloader=dataloader,
    )

    autoenc.eval()
    if cuda:
        autoenc.cuda()

    curses.noecho()
    curses.curs_set(False)
    pad = curses.newpad(80, 80)

    rows, cols = stdscr.getmaxyx()
    assert rows >= 64 and cols >= 64, "Terminal size needs to be at least 64x64"

    next_frame = time.time() + 1 / args.frame_rate
    embedding2 = get_random(device, dataset, autoenc.encoder)
    while True:
        embedding1 = embedding2
        embedding2 = get_random(device, dataset, autoenc.encoder)

        for x in np.linspace(0, 1, args.steps):

            if time.time() > next_frame:
                next_frame = time.time() + 1 / args.frame_rate

            x_scaled = np.log10(x ** args.smooth_factor + 1) * 3.322
            with torch.no_grad():
                interp_embedding = x_scaled * embedding2 + (1 - x) * embedding1
                decoded = autoenc.decoder(interp_embedding)
                decoded_str = ascii_util.one_hot_embedded_matrix_to_string(decoded[0])


            pad.addstr(0, 0, decoded_str)
            pad.refresh(0, 0, 0, 0, 64, 64)

        time.sleep(args.hold_length)


def get_random(device, dataset, encoder):
    img, label = dataset[random.randint(0, len(dataset) - 1)]
    img = torch.FloatTensor(img).to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        embedding, logvar = encoder(img)
    return embedding


if __name__ in {"__main__", "__console__"}:
    args = get_args()
    curses.wrapper(main, args)
