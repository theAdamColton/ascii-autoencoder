"""
Train an autoencoder on the ascii dataset
"""
import bpdb
import random
import numpy as np
import torch
import argparse
import torch.nn as nn
import os.path as path
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import bpdb


from dataset import AsciiArtDataset
import utils
import ascii_util
from autoencoder_models import VanillaAutoenc, VanillaDisc


def main():
    global args
    parser = argparse.ArgumentParser()

    # Arguments that don't require reinitializing the model
    parser.add_argument("--print-every", "-p", dest="print_every", default=10, type=int)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="autoenc",
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
        help="Number of dataset workers, not compatible with --keep-training-data-on-gpu",
    )
    parser.add_argument("--learning-rate", dest="learning_rate", default=5e-5, type=float)
    parser.add_argument("--latent-noise", dest="latent_noise", default=None, type=float, help="std of noise added to latent space for training reconstruction loss.")
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
    parser.add_argument(
        "--keep-training-data-on-gpu",
        dest="keep_training_data_on_gpu",
        type=bool,
        default=False,
        help="Speed up training by precaching all training data on the gpu",
    )
    parser.add_argument("-l", "--load", dest="load", help="load models from directory")

    parser.add_argument("-r", "--res", dest="res", type=int, default=32)
    parser.add_argument("--one-hot", dest="one_hot", action="store_true", default=False)
    parser.add_argument("--validation-prop", dest="validation_prop", default=None, type=float)
    parser.add_argument("--char-dim", dest="char_dim", type=int, default=8)
    parser.add_argument("--nz", dest="nz", type=int, default=None)
    parser.add_argument("--adversarial", action="store_true", dest="adversarial", help="Enable training of a discriminator from random samples from the z latent space")

    # Adversarial specific arguments
    parser.add_argument("--train-disc-every", type=int, dest="train_disc_every", default=1)
    parser.add_argument("--train-gen-every", type=int, dest="train_gen_every", default=1)
    parser.add_argument("--train-autoenc-every", type=int, dest="train_autoenc_every", default=1)

    args = parser.parse_args()

    # Argument correctness
    if args.one_hot:
        embedding_kind = "one-hot"
        channels=95
    else:
        embedding_kind = "decompose"
        channels=args.char_dim

    if not args.nz:
        z_dim = args.res * args.res
    else:
        z_dim = args.nz

    dataset = AsciiArtDataset(
        res=args.res,
        embedding_kind=embedding_kind,
        should_min_max_transform=not args.one_hot,
        channels=channels,
        validation_prop=args.validation_prop
    )
    if args.keep_training_data_on_gpu:
        tdataset = dataset.to_tensordataset(torch.device("cuda"))
    else:
        tdataset = dataset
    dataloader = DataLoader(
        tdataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=not args.keep_training_data_on_gpu,
    )

    # Vanilla Autoencoder
    autoenc = VanillaAutoenc(n_channels=channels, z_dim=z_dim)
    autoenc.cuda()
    bce_loss = nn.BCELoss()
    bce_loss.cuda()

    # The discriminator
    if args.adversarial:
        discriminator = VanillaDisc(args.nz)
        discriminator.cuda()

    # weights for ce loss
    char_weights = torch.zeros(95)
    # Less emphasis on space characters
    reduction_factor = 0.95
    char_weights[0] = 1 / 95 / reduction_factor
    char_weights[1:] = (1 - char_weights[0]) / 94
    ce_loss = nn.CrossEntropyLoss(weight=char_weights)
    ce_loss.cuda()

    mse_loss = nn.MSELoss()
    mse_loss.cuda()

    optimizer = torch.optim.Adam(autoenc.parameters(), lr=args.learning_rate)
    if args.adversarial:
        optim_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    Tensor = torch.cuda.FloatTensor
    device = torch.device("cuda")

    if args.load:
        autoenc.load_state_dict(
            torch.load(path.join(args.load, "autoencoder.pth.tar"))
        )
        if args.adversarial:
            discriminator.load_state_dict(
                torch.load(path.join(args.load, "discrim.pth.tar"))
            )
        print("Loaded autoencoder")
        with open(path.join(args.load, "epoch"), "r") as f:
            start_epoch = int(f.read())
    else:
        start_epoch = 0

    in_tensor = torch.rand(7, channels, args.res, args.res)
    in_tensor = in_tensor.to(device)
    print("Encoder:")
    out_tensor = utils.debug_model(autoenc.encoder, in_tensor)
    print("Decoder:")
    utils.debug_model(autoenc.decoder, out_tensor)

    if args.adversarial:
        print("Discriminator")
        in_tensor = torch.rand(7, args.nz)
        in_tensor = in_tensor.to(device)
        utils.debug_model(discriminator, in_tensor)

    g_loss = None
    d_loss = None

    for epoch in tqdm(range(start_epoch, args.n_epochs)):
        for i, data in enumerate(dataloader):
            images = data[0]
            images = Variable(images.type(Tensor))
            labels = data[1]

            if epoch % args.train_autoenc_every == 0:
                # Vanilla Autoencoder loss
                optimizer.zero_grad()
                z = autoenc.encoder(images)
                if args.latent_noise:
                    noise = Tensor(torch.randn(*z.shape, device=device) * args.latent_noise)
                    z += noise
                gen_im = autoenc.decoder(z)

                if args.one_hot:
                    loss = ce_loss(gen_im, images.argmax(1))
                else:
                    loss = bce_loss(gen_im, images)

                loss.backward()
                optimizer.step()

            if args.adversarial:
                if epoch % args.train_disc_every == 0:
                    autoenc.eval()
                    discriminator.zero_grad()
                    # Discriminator loss
                    # Sample from N(0, 5)
                    z_real_gauss = Variable(torch.randn(*z.shape, device=device) * 5)
                    d_real_gauss = discriminator(z_real_gauss)
                    z_fake_gauss = autoenc.encoder(images)
                    d_fake_gauss = discriminator(z_fake_gauss)

                    real_label = torch.full(d_real_gauss.shape, 1.0, dtype=torch.float, device=device, requires_grad=False)
                    fake_label = torch.full(d_fake_gauss.shape, 0.0, dtype=torch.float, device=device, requires_grad=False)

                    real_loss = bce_loss(d_real_gauss, real_label)
                    fake_loss = bce_loss(d_fake_gauss, fake_label)
                    d_loss = (real_loss + fake_loss) /2
                    d_loss.backward()
                    optim_D.step()
                    autoenc.train()

                if epoch % args.train_gen_every == 0:
                    autoenc.zero_grad()
                    # Generator loss
                    z_fake_gauss = autoenc.encoder(images)
                    d_fake_gauss = discriminator(z_fake_gauss)
                    
                    real_label = torch.full((images.shape[0], 1), 1.0, dtype=torch.float, device=device, requires_grad=False)
                    g_loss = bce_loss(d_fake_gauss, real_label)
                    g_loss.backward()
                    optimizer.step()


        if epoch % args.print_every == 0:
            # Preview images from validation data if it was set as a flag
            if args.validation_prop:
                image, label = dataset.get_validation_item(random.randint(0, dataset.get_validation_length()-1))
            else:
                image, label = dataset[random.randint(0,len(dataset)-1)]
            with torch.no_grad():
                autoenc.eval()
                gen_im = autoenc(Tensor(image).unsqueeze(0))
                autoenc.train()
            image_str = dataset.decode(image)
            gen_str = dataset.decode(gen_im[0].detach())
            side_by_side = ascii_util.horizontal_concat(image_str, gen_str)
            print(side_by_side)
            print(label)

        print("Epoch [{}/{}] Recon. loss: {}".format(epoch,args.n_epochs, loss.item()/args.batch_size), end="")
        if args.adversarial:
            if d_loss is not None:
                print(" Disc. loss {}".format(d_loss.item()), end="")
            if g_loss is not None:
                print(" Gen. loss {}".format(g_loss.item()), end="")
        print()

        if epoch % args.save_every == 0:
            if args.adversarial:
                save(autoenc, epoch, "./models/{}/".format(args.run_name), discriminator=discriminator)
            else:
                save(autoenc, epoch, "./models/{}/".format(args.run_name))


    if args.adversarial:
        save(autoenc, epoch, "./models/{}/".format(args.run_name), discriminator = discriminator)
    else:
        save(autoenc, epoch, "./models/{}/".format(args.run_name))


def save(autoenc: VanillaAutoenc, epoch: int, models_dir: str, discriminator = None):
    os.makedirs(models_dir, exist_ok=True) 
    torch.save(autoenc.state_dict(), "{}/autoencoder.pth.tar".format(models_dir))
    if discriminator:
        torch.save(discriminator.state_dict(), "{}/discrim.pth.tar".format(models_dir))
    with open(path.join(models_dir, "epoch"), "w") as f:
        f.write(str(epoch))
    print("saved.")

if __name__ in {"__main__", "__console__"}:
    main()
