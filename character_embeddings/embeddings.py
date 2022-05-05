"""
Utilities for dealing with embedding and de embedding characters
"""

import numpy as np
import torch
from os import path
from scipy.spatial import cKDTree


class CharacterEmbeddings:
    """
    Character embeddings for ascii chars from 32 to 126, inclusive
    """

    def __init__(self, ce_path=None):
        if not ce_path:
            ce_path = path.abspath(path.dirname(__file__) + "/character_embeddings.npy")
        self.embeddings = np.load(ce_path)
        self.char_dim = self.embeddings.shape[1]
        self.min = self.embeddings.min()
        self.max = self.embeddings.max()

    def embed(self, s: str) -> np.ndarray:
        """
        Returns a 2 by len(s) tensor with the character embedding for each char
        in s
        """
        out = np.zeros([len(s), self.char_dim])
        for i, c in enumerate(s):
            out[i] = self.embed_char(c)

        return out

    def embed_char(self, c: str) -> np.ndarray:
        ascii_code = ord(c) - 32
        return self.embeddings[ascii_code]

    def de_embed(self, a: np.ndarray) -> str:
        """
        a is a N_chars by self.char_dim array
        returns a length N_chars string
        """
        out = ""
        for char_embedding_vec in a:
            c = self.de_embed_char(char_embedding_vec)
            out += c
        return out

    def de_embed_char(self, c: np.ndarray) -> str:
        """
        Matches c to the nearest neighbor in self.embeddings
        c is a imdim **2 by char_dim array
        """
        idx_nearest_neighbor = cKDTree(self.embeddings).query(c, k=1)[1]
        ascii_code = idx_nearest_neighbor + 32
        return chr(ascii_code)

    def inverse_min_max_scaling(self, c: np.ndarray) -> np.ndarray:
        """
        Inverses min_max_scaling based on the max and min embeddings
        c is a char_dim by imdim by imdim array
        """
        c = c * (self.max - self.min) + self.min
        return c
