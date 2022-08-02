from pytorch_lightning.callbacks import StochasticWeightAveraging
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import bpdb
from os import path
import sys


dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./ascii-dataset/"))
from dataset import AsciiArtDataset

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./neural-font-renderer/"))
from lightning_model import PLNeuralRenderer

dirname = path.dirname(__file__)
sys.path.insert(0, path.join(dirname, "./python-pytorch-font-renderer/"))
from font_renderer import FontRenderer

from autoenc_trainers import LightningOneHotVAE


def get_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--kl-coeff", dest="kl_coeff", type=float, default=1.0)
    parser.add_argument(
        "--datapath",
        dest="datapath",
        default=None,
        help="Useful for memory-pinned data directories in /dev/shm/",
    )
    parser.add_argument(
        "--should-discrete-renderer",
        dest="should_discrete_renderer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--neural-renderer-path", dest="neural_renderer_path", default=None
    )
    parser.add_argument(
        "--ce-recon-loss-scale", dest="ce_recon_loss_scale", default=0.1, type=float
    )
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
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of dataset workers",
    )
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
        help="If this argument is close to zero, the char weights will be weighed more similarly.",
    )

    args = parser.parse_args()
    if args.should_discrete_renderer:
        assert not args.neural_renderer_path
    else:
        assert args.neural_renderer_path

    return args


def main():
    args = get_training_args()

    dataset = AsciiArtDataset(
        res=64, validation_prop=args.validation_prop, datapath=args.datapath
    )
    validation_dataset = AsciiArtDataset(
        res=64, validation_prop=args.validation_prop, is_validation_dataset=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
    )

    if args.neural_renderer_path:
        font_renderer = PLNeuralRenderer.load_from_checkpoint(args.neural_renderer_path)
        font_renderer.eval()
    else:
        font_renderer = FontRenderer(res=16, device=torch.device("cuda"))

    if args.should_char_weight:
        character_frequencies = dataset.calculate_character_counts()
        char_weights = 1.0 / (character_frequencies + 1)
        char_weights = char_weights**args.char_weights_scaling
    else:
        char_weights = torch.ones(95)

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
        )
        vae.init_weights()

    else:
        vae = LightningOneHotVAE.load_from_checkpoint(args.load, font_renderer=font_renderer, train_dataloader=dataloader, val_dataloader=val_dataloader)
        vae.font_renderer = font_renderer
        vae.lr = args.learning_rate
        vae.print_every = args.print_every
        vae.ce_recon_loss_scale = args.ce_recon_loss_scale
        vae.image_recon_loss_coeff = args.image_recon_loss_coeff
        vae.kl_coeff = args.kl_coeff
        vae.ce_loss = torch.nn.CrossEntropyLoss(weight=char_weights)
        print("Resuming training")

    logger = pl.loggers.TensorBoardLogger(args.run_name + "checkpoint/")

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
        precision=16,
        default_root_dir=args.run_name + "checkpoint/",
        callbacks=[StochasticWeightAveraging()],
        check_val_every_n_epoch=5,
        auto_lr_find=True,
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model=vae, train_dataloaders=dataloader, val_dataloaders=val_dataloader)


if __name__ in {"__main__", "__console__"}:
    main()
