import argparse
import os
import pathlib
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchdatasets as td
import torch.nn as nn
from torchsummary import summary

import bpdb

from dcganmodels import DCGAN_D, DCGAN_G, DLinGan_D, DLinGan_G
from character_embeddings.embeddings import CharacterEmbeddings
from dataset import AsciiArtDataset
import utils


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")

    """ Settings that do not require resetting the model """
    parser.add_argument(
        "-n",
        "--n_epochs",
        dest="n_epochs",
        default=200,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument("--lr", dest="lr", default=7E-5)
    parser.add_argument(
        "--keep-training-data-on-gpu",
        dest="keep_training_data_on_gpu",
        type=bool,
        default=False,
        help="Speed up training by precaching all training data on the gpu",
    )
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=0,
        help="Number of dataset workers, not compatible with --keep-training-data-on-gpu",
    )
    parser.add_argument(
        "-s",
        "--save-every",
        dest="save_every",
        type=int,
        help="save every n epochs",
        default=100,
    )
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="test_run",
    )
    parser.add_argument(
        "--autoenc-path", dest="autoenc_path"
    )
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")
    parser.add_argument(
        "--train-gen-every", dest="train_gen_every", type=int, default=1
    )
    parser.add_argument(
        "--train-dis-every", dest="train_dis_every", type=int, default=1
    )
    parser.add_argument("--noise-std", dest="noise_std", type=float, default=0.00)

    """ Settings that require recreating the model """
    parser.add_argument("-r", "--res", dest="res", type=int)
    parser.add_argument("--autoenc-latent-dim", dest="autoenc_latent_dim", type=int)
    parser.add_argument("--nz", dest="nz", type=int)
    args = parser.parse_args()

    # Arguments validity checking
    assert args.train_gen_every == 1 or args.train_dis_every == 1

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    nz = args.nz
    lr = args.lr
    b1 = 0.5
    b2 = 0.99

    img_size = args.res
    autoenc_latent_dim = args.autoenc_latent_dim

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _kwargs = dict(
        res=img_size,
        embedding_kind='one-hot',
    )
    dataset = AsciiArtDataset(
        **{k: v for k, v in _kwargs.items() if v is not None},
    )

    if args.keep_training_data_on_gpu:
        tdataset = dataset.to_tensordataset(device=device)
    else:
        tdataset = dataset

    dataloader = DataLoader(
        tdataset,
        batch_size=batch_size,
        num_workers=args.n_workers,
        shuffle=True,
        pin_memory=not args.keep_training_data_on_gpu,
    )

    run_dir = os.path.join(os.path.dirname(__file__), "models/", args.run_name)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    generator = DLinGan_G(nz, autoenc_latent_dim, )
    discriminator = DLinGan_D(autoenc_latent_dim, )
    generator.apply(utils.weights_init)
    discriminator.apply(utils.weights_init)

    # Load state dicts
    start_epoch = 0
    if args.load:
        models = [generator, discriminator]
        for model in models:
            model.load_state_dict(
                torch.load(os.path.join(args.load, model.name + ".pth.tar"))
            )
        print("Loaded models")
        with open(os.path.join(args.load, "epoch"), "r") as f:
            start_epoch = int(f.read())

    if cuda:
        generator.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer_GE = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    __noise = torch.randn(nz, device=device)
    print("-----Generator-----")
    summary(generator, (nz,))
    print("-------------------")
    print("----Discriminator--")
    summary(
        discriminator,
        (autoenc_latent_dim,)
    )
    print("-------------------")

    d_loss, g_loss = None, None

    # Training loop
    print("\nBegin training session with %i epochs...\n" % (n_epochs))
    for epoch in range(start_epoch, n_epochs):
        for imgs in dataloader:
            imgs = imgs[0]

            # Adversarial ground truths
            valid = Variable(
                Tensor(imgs.shape[0],).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                Tensor(imgs.shape[0],).fill_(0.0), requires_grad=False
            )

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # Sample random latent space
            noise = torch.randn(imgs.shape[0], nz, device=device)

            # Generate a batch of images
            gen_imgs = generator(noise)

            # -----------------
            #  Train Generator
            # -----------------

            if epoch % args.train_gen_every == 0:
                optimizer_GE.zero_grad()
                # Loss measures generator's ability to fool the discriminator
                g_loss = bce_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if epoch % args.train_dis_every == 0:
                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = bce_loss(discriminator(real_imgs), valid)
                fake_p = discriminator(gen_imgs.detach())
                fake_loss = bce_loss(fake_p, fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

        if d_loss is not None and g_loss is not None:
            print(
                "[Epoch %d/%d] \n"
                "\tModel Losses: [D: %f] [GE: %f]"
                % (epoch, n_epochs, d_loss.item(), g_loss.item())
            )

        if epoch % args.save_every == 0:
            save(discriminator, generator, run_dir, epoch)

        if epoch % args.print_every == 0:
            # Sample random latent space
            noise = torch.randn(1, nz, device=device)
            with torch.no_grad():
                generator.eval()
                gen_im = generator(noise)[0]
                real_im = dataset[random.randint(0, len(dataset))][0]
                #print(dataset.decode(real_im))
                #print(dataset.decode(gen_im[0]))
            generator.train()

    # Save current state of trained models
    save(discriminator, generator, run_dir, n_epochs)


def save(disc, gen, models_dir, epoch):
    print("Saving... {}".format(models_dir))
    os.makedirs(models_dir, exist_ok=True)
    model_list = [disc, gen]
    utils.save_model(models=model_list, out_dir=models_dir)
    with open(os.path.join(models_dir, "epoch"), "w") as f:
        f.write(str(epoch))


if __name__ in {"__main__", "__console__"}:
    main()
