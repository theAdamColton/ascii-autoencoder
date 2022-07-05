# Code from
# https://github.com/zhampel/clusterGAN/

import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F



import ascii_util


def tensor_to_ascii_decomp(sample, dataset, img_size, channels):
    """
    Takes a tensor encoded with pca character embeddings
    returns the string representation
    """
    if not type(sample) == np.ndarray:
        sample = sample.cpu()
    sample_rescaled = dataset.character_embeddings.inverse_min_max_scaling(sample)
    sample_rescaled = sample_rescaled.string_reshape(img_size ** 2, channels)
    s = dataset.character_embeddings.de_embed(sample_rescaled)
    s_res = ascii_util.string_reshape(s, img_size)
    return s_res


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


def sample_gumbel(shape, eps=1e-20):
    """Samples from the Gumbel distribution
    Returns a shape tensor from the Gumbel distribution
    """
    U = torch.rand(shape)
    U=U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, dim=-1):
    """Samples from the Gumbel distribution, and then softmaxs to produce one hot encoded 
    vectors along dimension dim

    (The return tensor will sum to one along dimension <dim>)
    """
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False, dim=-1):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: [*, n_class] an one-hot vector along dimension <dim>
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    
    if not hard:
        #return y.view(-1, latent_dim * categorical_dim)
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    #return y_hard.view(-1, latent_dim * categorical_dim)
    return y_hard

