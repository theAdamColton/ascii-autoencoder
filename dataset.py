import ascii_util
from character_embeddings.embeddings import CharacterEmbeddings

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from os import path
from glob import glob

DATADIR = path.abspath(path.join(path.dirname(__name__), "data_aggregation/data/"))
import utils


class AsciiArtDataset(Dataset):
    """
    Initialize an AsciiArtDataset, from a directory structure where subfolders are the sub categories
    And ascii art files are stored as .txt ascii files. If `res` is passed, all arts larger (greater #
    of lines, or greater # of columns) than this value are not included in the dataset. Ascii files
    smaller than res will be padded with spaces.

    Ascii art files in the datapath are expected to already be pre padded to the right with zeros to
    the length of the maximum width line.

    Files are returned as tensors, with each character being represented by a vector embedding.
    These embedding can be obtained from PCA, shown in `./character_embeddings/`,
    or these embeddings can be obtained as 'one-hot' vectors of length 95.
    The only characters that are allowed are 32 to 126 (inclusive).
    """

    def __init__(
        self,
        res: int = 36,
        datapath=DATADIR,
        embedding_kind="decompose",
        should_min_max_transform=False,
        channels=8,
        max_samples=None,
        validation_prop=None
    ):
        """
        res: Desired resolution of the square ascii art
        datapath: Optional specification of the directory containing *.txt files, organized by directory in categories
        embedding_kind: One of 'one-hot', 'decompose'
            'one-hot' indicates that each character in the image will be represented by a
        should_min_max_transform: If true, will scale all the data to 0 - 1, by examining the smallest and largest values
        channels: number of channels to use if embedding_kind is decompose.
        max_samples: The maximum number of training samples to take.
        validation_prop: The proportion of data to use as validation
        """
        self.res = res
        self.should_min_max_transform = should_min_max_transform
        self.embedding_kind = embedding_kind

        assert self.embedding_kind in {"decompose", "one-hot"}
        if self.should_min_max_transform:
            assert self.embedding_kind == "decompose"

        if self.embedding_kind == "decompose":
            character_embeddings = path.join(
                path.dirname(__file__),
                "./character_embeddings/character_embeddings{}d.npy".format(channels),
            )
            self.character_embeddings = CharacterEmbeddings(character_embeddings)
            self.channels = self.character_embeddings.char_dim
        elif self.embedding_kind == "one-hot":
            self.channels = 95

        if should_min_max_transform:
            self.min_char_emb = self.character_embeddings.min
            self.max_char_emb = self.character_embeddings.max

        assert path.isdir(datapath)
        # Filters out files that are too large
        asciifiles = set(glob(path.join(datapath, "**/*.txt"), recursive=True))
        for file in list(asciifiles).copy():
            with open(file, "r") as f:
                line_count = sum(1 for _ in f)
            with open(file, "r") as f:
                line1 = f.readline()
                # The -1 is because we don't count \n's
                line_width = len(line1.replace("\n", ""))
            if res is not None:
                if line_width > res or line_count > res:
                    asciifiles.remove(file)
                    # print("popped {}, too big".format(file))
                    continue

            with open(file, "r") as f:
                for line in f:
                    for s in line:
                        # Only characters in 10, [32, 126] are allowed
                        code = ord(s)
                        if code != 10 and (code < 32 or code > 126):
                            if file in asciifiles:
                                asciifiles.remove(file)

        self.asciifiles = list(asciifiles)
        if validation_prop:
            max_idx = int(len(self.asciifiles) * validation_prop)
            self.validation_ascii_files = self.asciifiles[max_idx:]
            self.asciifiles = self.asciifiles[0:max_idx]
        if max_samples:
            self.asciifiles = self.asciifiles[:max_samples+1]

    def __len__(self):
        return len(self.asciifiles)

    def __getitem__(self, index):
        """
        Returns the character_embeddings representation of the string,
        as a self.channels by self.res by self.res array
        """
        filename = self.asciifiles[index]
        return self.__getitem_from_filename__(filename)

    def get_validation_item(self, index):
        filename = self.validation_ascii_files[index]
        return self.__getitem_from_filename__(filename)

    def get_validation_length(self):
        return len(self.validation_ascii_files)

    def __getitem_from_filename__(self, filename):
        with open(filename, "r") as f:
            content = f.read()

        content = ascii_util.raw_string_to_squareized(content, self.res)
        
        # Embeds characters
        if self.embedding_kind == "decompose":
            embeddings = self.character_embeddings.embed(content)
            if self.should_min_max_transform:
                # Min max scaling
                embeddings = (embeddings - self.min_char_emb) / (
                    self.max_char_emb - self.min_char_emb
                )
            # Makes embeddings image_res by image_res by channel
            embeddings = embeddings.reshape(self.res,self.res,self.channels)
            # Makes embeddings nchannels by image_res by image_res
            embeddings = np.moveaxis(embeddings, 2,0)

        elif self.embedding_kind == "one-hot":
            embeddings = ascii_util.squareized_string_to_one_hot(content, self.res)


        label = self.__get_category_string_from_datapath(filename)

        return embeddings, label

    def to_tensordataset(self, device) -> TensorDataset:
        out = torch.Tensor(
            len(self),
            self.channels,
            self.res,
            self.res,
        ).to(device)
        for i in range(len(self)):
            out[i] = torch.Tensor(self[i][0]).to(device)
        return TensorDataset(out)

    def get_all_category_strings(self):
        """Returns all category strings,unordered"""
        d = set()
        for x in self.asciifiles:
            d.add(self.__get_category_string_from_datapath(x))
        return list(d)

    def __get_category_string_from_datapath(self, datapath: str) -> str:
        return utils.remove_prefix(path.dirname(datapath), DATADIR)

    def decode(self, x) -> str:
        """Takes a matrix of character embeddings, returns a string with correct line breaks"""

        if not type(x) == np.ndarray:
            x = x.cpu()
            x = np.array(x)
        if self.should_min_max_transform:
            x = self.character_embeddings.inverse_min_max_scaling(x)

        if self.embedding_kind == "one-hot":
            return ascii_util.one_hot_embedded_matrix_to_string(x)
        elif self.embedding_kind == "decompose":
            # Moves channels to last dim
            x = np.moveaxis(x, 0, 2)
            # Reshapes
            x = x.reshape(self.res**2, self.channels)

            s = self.character_embeddings.de_embed(x)
            s_res = ascii_util.string_reshape(s, self.res)
            return s_res

    def get_file_name(self, i):
        return self.asciifiles[i]
