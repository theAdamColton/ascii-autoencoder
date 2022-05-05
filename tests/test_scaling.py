import unittest
import os
import bpdb

import ascii_util

from character_embeddings.embeddings import CharacterEmbeddings
from dataset import AsciiArtDataset

class TestScaling(unittest.TestCase):
    def test_inverse(self):
        ds = AsciiArtDataset(datapath=os.path.dirname(__file__) + "/../data_aggregation/data/", should_min_max_transform=True, res=36)
        before = ds[0][0]
        after = ds.character_embeddings.inverse_min_max_scaling(before)
        print(after)
        after = after.string_reshape(ds.res ** 2, 8)
        res = ds.character_embeddings.de_embed(after)
        res = ascii_util.string_reshape(res, ds.res)
        print(res)

