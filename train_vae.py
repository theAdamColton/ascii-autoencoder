import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse

from dataset import AsciiArtDataset
import utils
from autoencoder_models import OneHotVariationalAutoEncoder, VAELoss
from autoenc_trainers import LightningOneHotVAE


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

    # Hyperparams
    parser.add_argument("--anneal-rate", dest="anneal_rate", default=3e-5, type=float)
    parser.add_argument("--temp-min", dest="temp_min", default=0.5, type=float)

    parser.add_argument("--validation-prop", dest="validation_prop", default=None, type=float)
    #parser.add_argument("--nz", dest="nz", help="z dimension", type=int, default=128)

    args = parser.parse_args()
    return args


def main():
    args = get_training_args()

    # Hardcoded one hot vector length
    n_channels = 95
    # Hardcoded nz
    nz = 128

    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = AsciiArtDataset(res=64, embedding_kind='one-hot', validation_prop=args.validation_prop)
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=True,
    )

    if not args.load:
        vae = OneHotVariationalAutoEncoder(n_channels, nz, device)
        vae_loss = VAELoss()
        lit_vae = LightningOneHotVAE(vae, vae_loss, lr=args.learning_rate, print_every=args.print_every)

        lit_vae = lit_vae.to(torch.double)
    else:
        lit_vae = LightningOneHotVAE.load_from_checkpoint(args.load)
        print("Resuming training")

    if cuda:
        lit_vae.cuda()

    # The 'period' argument changes for different versions of pytorch lightning, in newer versions it is 'every_n_epochs'
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="vae_models/", period = args.save_every, save_last=True)

    trainer = pl.Trainer(max_epochs=args.n_epochs, accelerator='tpu', gpus=-1, default_root_dir="vae_checkpoint/")

    trainer.fit(model=lit_vae, train_dataloader=dataloader)


if __name__ in {"__main__", "__console__"}:
    main()
