import unittest
import dataset

class TestDataloader(unittest.TestCase):
    def test_get_all_ascii(self):
        dl = dataset.AsciiArtDataset(res=60)
        size = None
        for i in range(len(dl)):
            res, label = dl[i]
            if size is None:
                size = res.shape
                print(size)
            else:
                if res.shape != size:
                    print(i)
                    print(dl.get_file_name(i))
                self.assertEqual(res.shape,  size)

