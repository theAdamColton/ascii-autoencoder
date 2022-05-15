"""
For each file, gets the maximum line length, pads the end of the line to the
max line length with zeros, and then writes it.
"""

import glob
import subprocess


def get_max_line_length(file) -> int:
    max_len = 0
    for line in file.readlines():
        l = len(line)
        if l > max_len:
            max_len = l
    return max_len


def pad_line(line: str, length: int) -> str:
    return line.ljust(length, " ")


txtfiles = glob.glob("data/**/*.txt", recursive=True)

# Gets max lengths
print("Writing tmp files")
max_line_lengths = [0]*len(txtfiles)
for i, filename in enumerate(txtfiles):
    with open(filename, "r") as f:
        max_line_lengths[i] = get_max_line_length(f)

# Writes temp files
for i, filename in enumerate(txtfiles):
    with open(filename, "r") as f, open(filename + ".tmp", "w") as fout:
        max_line = max_line_lengths[i] -1
        for line in f.readlines():
            line = line.replace("\n", "")
            padded_line = pad_line(line, max_line)
            fout.write(padded_line + "\n")

# Moves temp files
print("Moving tmp files")
for tempfilename in glob.glob("data/**/*.txt.tmp", recursive=True):
    pass
    subprocess.run(["mv", tempfilename, tempfilename.removesuffix(".tmp")])
