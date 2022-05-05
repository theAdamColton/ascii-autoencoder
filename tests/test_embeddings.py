import unittest
import character_embeddings.embeddings
import numpy as np


class TestEmbeddings(unittest.TestCase):
    def get_ce(self):
        return character_embeddings.embeddings.CharacterEmbeddings()

    def test_embed(self):
        ce = self.get_ce()
        res = ce.embed("Hello There")
        print(res)
        res = ce.embed("|/|/|/|/|/|/")
        print(res)

    def test_embed_and_de_embed_rand(self):
        ce = self.get_ce()
        rand_ascii = np.random.randint(32,126, size=10)
        rand_string = "".join(chr(ac) for ac in rand_ascii)
        embed = ce.embed(rand_string)
        res = ce.de_embed(embed)
        self.assertEqual(res, rand_string)


