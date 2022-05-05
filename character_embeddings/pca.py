"""
Performs PCA on all of the character images in out/
This is for the purpose of obtaining character embeddings based on pixel appearance of the characters
"""

## https://medium.com/@sebastiannorena/pca-principal-components-analysis-applied-to-images-of-faces-d2fc2c083371
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os
import bpdb


imres = 64  # 64x64
font_im_dir = os.path.abspath(os.path.dirname(__file__) + "/out/")
imfiles = os.listdir(font_im_dir)


def demonstration():
    embeddings, projected = generate_character_embeddings(16)

    if input("Show images?").startswith("y"):
        for i, proj in enumerate(projected):
            im = proj.string_reshape(imres, imres)
            plt.imshow(im, cmap="gray")
            imnum = int(imfiles[i].removesuffix(".png"))
            plt.title("im {} char '{}'".format(imnum, chr(imnum + 32)))
            plt.show()

    embeddings_2d, _ = generate_character_embeddings(2)

    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title("2D character embeddings")

    for i in range(embeddings_2d.shape[0]):
        plt.text(
            x=embeddings_2d[i, 0] + 0.3,
            y=embeddings_2d[i, 1] + 0.3,
            s=chr(i + 32),
            fontdict={"size": 26},
        )
    plt.show()

    if input("save embeddings?").startswith("y"):
        with open("./character_embeddings.npy", "wb") as f:
            np.save(f, embeddings_2d)

    bpdb.set_trace()


def generate_character_embeddings(n_components=12):
    """
    Returns (embeddings, reprojected)
    embeddings are for ascii characters starting from 32, and going to 126
    To get an embeddings for an ascii character #x, do embeddings[x-32]
    """
    images = np.zeros([len(imfiles), 64, 64])

    for i, imfile in enumerate(imfiles):
        with Image.open(os.path.join(font_im_dir, imfile)) as im:
            im_arr = np.asarray(im)
            images[i] = im_arr

    images_flattened = images.reshape(images.shape[0], imres ** 2)
    p = PCA(n_components=n_components)
    images_pca = p.fit(images_flattened)

    transformed_comps = images_pca.transform(images_flattened)
    projected = images_pca.inverse_transform(transformed_comps)
    return (transformed_comps, projected)


if __name__ in {"__main__", "__console__"}:
    demonstration()
