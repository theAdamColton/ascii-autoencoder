# Code from
# https://github.com/zhampel/clusterGAN/

import numpy as np
from scipy import signal


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

