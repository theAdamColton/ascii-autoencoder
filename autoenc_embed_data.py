"""Creates embeddings for every txt file in the ascii art dir"""

import bpdb
import torch
import argparse
import os.path as path

from autoencoder_models import VanillaAutoenc
from dataset import AsciiArtDataset



def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")
    parser.add_argument("-r", "--res", dest="res", type=int, default=64)
    parser.add_argument("--nz", dest="nz", type=int, default=None)
    args = parser.parse_args()

    autoenc = VanillaAutoenc(n_channels=95, z_dim=args.nz)
    autoenc.cuda()
    autoenc.load_state_dict(
        torch.load(path.join(args.load, "autoencoder.pth.tar"))
    )
    autoenc.eval()
    print("Loaded autoencoder")

    dataset = AsciiArtDataset(res=args.res, embedding_kind='one-hot')
    files = dataset.asciifiles
    for i, (img, label) in enumerate(dataset):
        img = torch.cuda.FloatTensor(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            embedding = autoenc.encoder(img)
        filename = files[i]
        out_filename = filename.removesuffix(".txt") + ".pt"
        print("Saving {}".format(out_filename))
        torch.save(embedding, out_filename)

if __name__ in {"__main__", "__console__"}:
    main()
