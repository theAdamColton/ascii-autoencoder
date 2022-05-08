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
from torchsummary import summary
import torch.nn as nn
import bpdb


from dataset import AsciiArtDataset
import utils
from autoencoder_models import VAE


def main():
    global args
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--centralize-loss", dest="centralize_loss", default=False, help="Make the less greater for areas in the center of the image")
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
    parser.add_argument("--one-hot", dest="one_hot", type=bool, default=False)
    parser.add_argument("--char-dim", dest="char_dim", type=int, default=8)
    parser.add_argument("--nz", dest="nz", type=int, default=None)
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
        channels=channels
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

    vae = VAE(n_channels=channels, z_dim=z_dim)
    vae.cuda()
    bce_loss = nn.BCELoss()
    bce_loss.cuda()
    mse_loss = nn.MSELoss()
    mse_loss.cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    Tensor = torch.cuda.FloatTensor
    device = torch.device("cuda")

    if args.centralize_loss:
        loss_filter = utils.norm_gkern(args.res, 0.5)
        loss_filter = Tensor(loss_filter)

    if args.load:
        vae.load_state_dict(
            torch.load(path.join(args.load, "vae.pth.tar"))
        )
        print("Loaded vae")
        with open(path.join(args.load, "epoch"), "r") as f:
            start_epoch = int(f.read())
    else:
        start_epoch = 0

    in_tensor = torch.rand(7, channels, args.res, args.res)
    in_tensor = in_tensor.to(device)
    print("Encoder:")
    out_tensor = utils.debug_model(vae.encoder, in_tensor)
    print("Decoder:")
    utils.debug_model(vae.decoder, out_tensor)

    for epoch in range(start_epoch, args.n_epochs):
        for i, data in enumerate(dataloader):
            images = data[0]
            images = Variable(images.type(Tensor))

            labels = data[1]
            gen_im, mu, logvar = vae(images)
            loss = vae_loss(gen_im, images, mu, logvar, bce_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.print_every == 0:
            image, label = dataset[random.randint(0,len(dataset)-1)]
            with torch.no_grad():
                vae.eval()
                gen_im = vae(Tensor(image).unsqueeze(0))
                vae.train()
            print(dataset.decode(image))
            print(label)
            print(dataset.decode(gen_im[0].detach()))
            bpdb.set_trace()
        print("Epoch [{}/{}] Loss: {}".format(epoch,args.n_epochs, loss.item()/args.batch_size))

        if epoch % args.save_every == 0:
            print("Saving...")
            save(vae, epoch, "./models/{}/".format(args.run_name))

    save(vae, epoch, "./models/{}/".format(args.run_name))





def save(autoenc: VAE, epoch: int, models_dir: str):
    os.makedirs(models_dir, exist_ok=True) 
    torch.save(autoenc.state_dict(), "{}/vae.pth.tar".format(models_dir, epoch))
    with open(path.join(models_dir, "epoch"), "w") as f:
        f.write(str(epoch))

def vae_loss(recon_x, x, mu, logvar, loss_fn):
    l_loss = loss_fn(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return l_loss + kld



if __name__ in {"__main__", "__console__"}:
    main()
