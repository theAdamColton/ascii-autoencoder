from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
import torch
import torchinfo
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import bpdb
from os import path
import sys
import datetime


dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import ContinuousFontRenderer

from ascii_vae_trainer import LightningOneHotVAE


def get_training_args():
    parser = argparse.ArgumentParser()
    # Renderer args
    parser.add_argument(
        "--font-res",
        dest="font_res",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--font-zoom",
        dest="font_zoom",
        type=int,
        default=20,
    )

    # Loss coefficients
    parser.add_argument(
        "--ce-recon-loss-scale", dest="ce_recon_loss_scale", default=0.1, type=float
    )
    parser.add_argument("--kl-coeff", dest="kl_coeff", type=float, default=1.0)
    parser.add_argument(
        "--image-recon-loss-coeff",
        dest="image_recon_loss_coeff",
        default=1.0,
        type=float,
    )
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="vae",
    )

    # Dataset args
    parser.add_argument(
        "--datapath",
        dest="datapath",
        default=None,
        help="Useful for memory-pinned data directories in /dev/shm/",
    )
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of dataset workers",
    )
    parser.add_argument(
        "--dataset-to-gpu",
        dest="dataset_to_gpu",
        default=False,
        action="store_true",
    )

    # Training args
    parser.add_argument(
        "--learning-rate", dest="learning_rate", default=5e-5, type=float
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        default=64,
        type=int,
        help="Batch size",
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
    parser.add_argument(
        "--validation-prop", dest="validation_prop", default=None, type=float
    )
    parser.add_argument(
        "--validation-every", dest="validation_every", default=8, type=int
    )

    # Character weighting for CE loss
    parser.add_argument(
        "--should-char-weight",
        dest="should_char_weight",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--char-weights-scaling",
        dest="char_weights_scaling",
        default=0.1,
        type=float,
        help="If this argument is close to zero, the char weights will be weighed more similarly. When this argument is 1, (the default), the character weigts will be inversly proportional to their frequency in the dataset.",
    )
    parser.add_argument(
        "--space-deemph",
        dest="space_deemph",
        default=1.0,
        type=float,
        help="The space character weight is divided by this number.",
    )

    parser.add_argument(
        "--gumbel-tau",
        dest="gumbel_tau",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    return args


def main():
    args = get_training_args()

    dataset = AsciiArtDataset(
        res=64, validation_prop=args.validation_prop, datapath=args.datapath
    )
    validation_dataset = AsciiArtDataset(
        res=64, validation_prop=args.validation_prop, is_validation_dataset=True
    )

    if args.should_char_weight:
        character_frequencies = dataset.calculate_character_counts()
        char_weights = 1.0 / (character_frequencies + 1)
        char_weights = char_weights**args.char_weights_scaling
    else:
        char_weights = torch.ones(95)

    char_weights[0] = char_weights[0] / args.space_deemph

    print("Character weights: {}".format(char_weights))

    if args.dataset_to_gpu:
        args.n_workers = 0
        print("Loading data to gpu...")
        dataset = dataset.to_tensordataset(device=torch.device("cuda"))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=not args.dataset_to_gpu,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    # The character font size of each character in the image
    font_renderer = ContinuousFontRenderer(
        res=args.font_res, device=torch.device("cuda"), zoom=args.font_zoom,
    )

    if not args.load:
        vae = LightningOneHotVAE(
            font_renderer,
            dataloader,
            val_dataloader=val_dataloader,
            lr=args.learning_rate,
            print_every=args.print_every,
            char_weights=char_weights,
            ce_recon_loss_scale=args.ce_recon_loss_scale,
            image_recon_loss_coeff=args.image_recon_loss_coeff,
            kl_coeff=args.kl_coeff,
            gumbel_tau=args.gumbel_tau,
        )
        vae.init_weights(std=0.10)

    else:
        vae = LightningOneHotVAE.load_from_checkpoint(
            args.load,
            font_renderer=font_renderer,
            train_dataloader=dataloader,
            val_dataloader=val_dataloader,
        )
        vae.font_renderer = font_renderer
        vae.lr = args.learning_rate
        vae.print_every = args.print_every
        vae.ce_recon_loss_scale = args.ce_recon_loss_scale
        vae.image_recon_loss_coeff = args.image_recon_loss_coeff
        vae.kl_coeff = args.kl_coeff
        vae.ce_loss = torch.nn.CrossEntropyLoss(weight=char_weights)
        vae.gumbel_tau = args.gumbel_tau
        print("Resuming training")

    torchinfo.summary(vae.encoder, input_size=(7, 95, 64, 64))
    torchinfo.summary(vae.decoder, input_size=(7, 512))

    logger = pl.loggers.TensorBoardLogger(args.run_name + "checkpoint/")

    dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")

    model_checkpoint = ModelCheckpoint(
        dirpath="{}checkpoint/{}".format(args.run_name, dt_string),
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
#        precision=16,
        callbacks=[StochasticWeightAveraging(), model_checkpoint],
        check_val_every_n_epoch=args.validation_every,
        auto_lr_find=True,
        logger=logger,
        log_every_n_steps=10,
    )

    #trainer.tune(vae)

    trainer.fit(model=vae, train_dataloaders=dataloader, val_dataloaders=val_dataloader)


if __name__ in {"__main__", "__console__"}:
    main()
