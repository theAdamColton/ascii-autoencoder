# Code from
# https://github.com/zhampel/clusterGAN/

import os
import numpy as np
from scipy import signal

from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

from itertools import chain as ichain


import ascii_util


# Nan-avoiding logarithm
def tlog(x):
    return torch.log(x + 1e-8)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


# Cross Entropy loss with two vector inputs
def cross_entropy(pred, soft_targets):
    log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(-soft_targets * log_softmax_pred, 1))


# Save a provided model to file
def save_model(models=[], out_dir=""):

    # Ensure at least one model to save
    assert len(models) > 0, "Must have at least one model to save."

    # Save models to directory out_dir
    for model in models:
        filename = model.name + ".pth.tar"
        outfile = os.path.join(out_dir, filename)
        torch.save(model.state_dict(), outfile)


# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):

    assert fix_class == -1 or (fix_class >= 0 and fix_class < n_c), (
        "Requested class %i outside bounds." % fix_class
    )

    Tensor = torch.cuda.FloatTensor

    # Sample noise as generator input, zn
    zn = Variable(
        Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim))),
        requires_grad=req_grad,
    )

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if fix_class == -1:
        zc_idx = zc_idx.random_(n_c).cuda()
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.0)
        # zc_idx = torch.empty(shape, dtype=torch.long).random_(n_c).cuda()
        # zc_FT = Tensor(shape, n_c).fill_(0).scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda()
        zc_FT = zc_FT.cuda()

    zc = Variable(zc_FT, requires_grad=req_grad)

    ## Gaussian-noisey vector generation
    # zc = Variable(Tensor(np.random.normal(0, 1, (shape, n_c))), requires_grad=req_grad)
    # zc = softmax(zc)
    # zc_idx = torch.argmax(zc, dim=1)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def plot_train_loss(df=[], arr_list=[""], figname="training_loss.png"):

    fig, ax = plt.subplots(figsize=(16, 10))
    for arr in arr_list:
        label = df[arr][0]
        vals = df[arr][1]
        epochs = range(0, len(vals))
        ax.plot(epochs, vals, label=r"%s" % (label))

    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.set_title("Training Loss", fontsize=24)
    ax.grid()
    # plt.yscale('log')
    plt.legend(loc="upper right", numpoints=1, fontsize=16)
    print(figname)
    plt.tight_layout()
    fig.savefig(figname)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    raise Exception("No prefix")


def weights_init(m):
    """
    From the DCGAN paper, the authors specify that all
    model weights shall be randomly initialized from a
    Normal distribution with mean=0, stdev=0.02. The
    weights_init function takes an initialized model as
    input and reinitializes all convolutional,
    convolutional-transpose, and batch normalization
    layers to meet this criteria. This function is
    applied to the models immediately after
    initialization.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def tensor_to_ascii_decomp(sample, dataset, img_size, channels):
    """Takes a tensor encoded with pca character embeddings"""
    if not type(sample) == np.ndarray:
        sample = sample.cpu()
    sample_rescaled = dataset.character_embeddings.inverse_min_max_scaling(sample)
    sample_rescaled = sample_rescaled.string_reshape(img_size ** 2, channels)
    s = dataset.character_embeddings.de_embed(sample_rescaled)
    s_res = ascii_util.string_reshape(s, img_size)
    print(s_res)


def debug_model(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(m, output.shape)
    return output



def gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def norm_gkern(kernlen, std=1):
    """Returns a 2D Gaussian kernel array that sums to 1, with no negative elements"""
    k = gkern(kernlen, std)
    k /= k.sum()
    k[k<0] = 0
    return k
