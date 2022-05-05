import unittest
from ascii_util import pad_to_x_by_x
import os
from glob import glob


class TestPad(unittest.TestCase):
    def test_pad(self):
        s1 = "hello\njames\nfivel\nlette"
        s2 = "123\n456"
        p1 = pad_to_x_by_x(s1, 50)
        p2 = pad_to_x_by_x(s2, 50)
        self.assertEqual(len(p1), len(p2))

    def test_simple_pad(self):
        s1 = "a"
        s2 = "123"
        p1 = pad_to_x_by_x(s1, 3, char="*")
        p2 = pad_to_x_by_x(s2, 3, char="*")
        print("{}\n{}".format(repr(p1), repr(p2)))
        self.assertEqual(len(p1), len(p2), 9)
        self.assertEqual(len(p1.splitlines()[0]), 3)

    def test_data_pad(self):
        files = glob(os.path.dirname(__file__) + "/../data_aggregation/data/**/*.txt", recursive=True)
        pad_length = 80
        for file in files:
            with open(
                file,
                "r",
            ) as f:
                s = f.read()
            if len(s.splitlines()[0]) > pad_length:
                continue
            if len(s.splitlines()) > pad_length:
                continue
            p = pad_to_x_by_x(s, pad_length)
            lines = p.splitlines()
            length = len(lines)
            self.assertEqual(length, pad_length)
            for line in lines:
                self.assertEqual(length, len(line))
