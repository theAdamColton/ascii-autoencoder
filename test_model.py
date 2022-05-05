import argparse
from os import path
import torch
import bpdb
import numpy as np

from dcganmodels import DCGAN_D, DCGAN_G
from dataset import AsciiArtDataset
from character_embeddings.embeddings import CharacterEmbeddings
import utils
import ascii_util
import character_embeddings.embeddings


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model_path", help="model path")
    parser.add_argument("-p", dest="char_embed_path", help="path to .npy character embeddings array")
    parser.add_argument("-c", type=int, default=8, dest="channels")
    parser.add_argument("-l", dest="latent_dim", type=int)
    parser.add_argument("-nz", dest="nz", type=int)
    args = parser.parse_args()


    # Image sizes
    img_size = 20
    channels = args.channels

    if args.char_embed_path:
        char_embed_path = args.char_embed_path
    else:
        char_embed_path = path.dirname(__file__) + "/character_embeddings/character_embeddings8d.npy"

    # Dataset
    dataset = AsciiArtDataset(
        res=img_size,
        character_embeddings=char_embed_path
    )

    x_shape = (channels, img_size, img_size)

    generator = DCGAN_G(img_size, args.latent_dim, channels, args.latent_dim, args.nz, )
    discriminator = DCGAN_D(img_size, args.latent_dim, channels,  

    models = [generator, discriminator]
    for model in models:
        model.load_state_dict(
            torch.load(path.join(args.model_path, "models/", model.name + ".pth.tar"))
        )
    print("Loaded models")


    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Generates 
    with torch.no_grad():
        # Sample random latent variables
        zn, zc, zc_idx = utils.sample_z(
            # 10 samples
            shape=10, latent_dim=latent_dim, n_c=n_c
        )
        print("Generating...")
        gen_imgs = generator(zn, zc)

        for sample in gen_imgs:
            print("Converting to ascii...")
            utils.tensor_to_ascii_decomp(sample, dataset, img_size, channels)
            input("Next?")


if __name__ in {"__main__", "__console__"}:
    main()
