import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from dataset import AsciiArtDataset
import utils
from autoencoder_models import VariationalAutoEncoder, VAELoss
from autoenc_trainers import LightningVAE


def get_training_args():
    parser = argparse.ArgumentParser()

    # Arguments that don't require reinitializing the model
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="vae",
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
        "--n-workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of dataset workers",
    )
    parser.add_argument("--learning-rate", dest="learning_rate", default=5e-5, type=float)
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument(
        "-n",
        "--n_epochs",
        dest="n_epochs",
        default=200,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")

    parser.add_argument("--validation-prop", dest="validation_prop", default=None, type=float)
    parser.add_argument("--nz", dest="nz", help="z dimension", type=int, required=True)

    args = parser.parse_args()
    return args


def main():
    args = get_training_args()

    # Hardcoded one hot vector length
    n_channels = 95
    cuda = torch.cuda.is_available()

    dataset = AsciiArtDataset(res=64, embedding_kind='one-hot', validation_prop=args.validation_prop)
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=True,
    )

    vae = VariationalAutoEncoder(n_channels, args.nz)
    vae_loss = VAELoss()
    lit_vae = LightningVAE(vae, vae_loss, lr=args.learning_rate)

    if cuda:
        lit_vae.cuda()

    lit_vae = lit_vae.to(torch.double)

    trainer = pl.Trainer(max_epochs=args.n_epochs)

    trainer.fit(model=lit_vae, train_dataloader=dataloader)


if __name__ in {"__main__", "__console__"}:
    main()
