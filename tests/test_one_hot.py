import unittest

from character_embeddings import one_hot_encoding
from dataset import AsciiArtDataset


class TestOneHotEncoding(unittest.TestCase):
    def test_decompose_recompose(self):
        s = "Well, when the pipeline gets broken and I'm lost on the river bridge I'm cracked up on the highway and " \
            "on the water's edge She comes down the thruway ready to sew me up with thread Well, if I go down dyin', " \
            "you know she bound to put a blanket on my bed. "

        s_decomp = one_hot_encoding.get_one_hot_for_str(s)
        print(s_decomp)
        s_recomp = one_hot_encoding.one_hot_to_str(s_decomp)
        print(s_recomp)

        self.assertEqual(s, s_recomp)

    def test_one_hot_dataset(self):
        ds = AsciiArtDataset(res=20, embedding_kind='one-hot')
        print(len(ds))
        for x in ds:
            ds.decode(x[0])
