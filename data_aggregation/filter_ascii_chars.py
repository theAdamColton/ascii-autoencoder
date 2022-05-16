"""
Deletes all .txt files that have non allowed ascii characters
(not in [32 - 126])
"""

import subprocess
from glob import glob

datadir = "data/"

files = glob(datadir + "**/*.txt", recursive=True)

for file in files:
    with open(file, "r") as f:
        for line in f:
            for char in line:
                code = ord(char)
                if code == 10: continue
                if code > 300 or code < 32:
                    print("{}, {}, {}".format(file, char, code))
                    subprocess.run(['rm', 'f', file])
