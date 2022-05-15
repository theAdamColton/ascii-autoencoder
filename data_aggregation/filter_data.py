"""
Deletes files that are too small or too large
Also, deletes any files using tabs
"""

import os
import subprocess
import glob

txtfiles = glob.glob("data/**/*.txt", recursive=True)
lowerbytelimit = 25
bytelimit = 6000
nline_lowerlimit = 3
nline_upperlimit = 100

for file in txtfiles:
    delete = False

    nlines = int(
        subprocess.Popen(["wc", "-l", file], stdout=subprocess.PIPE)
        .communicate()[0]
        .split()[0]
    )
    if nlines < nline_lowerlimit:
        delete = True
        print("{} lines too small!".format(file))
    if nlines > nline_upperlimit:
        print("{}  lines too beaucoup".format(file))
        delete = True

    nbytes = os.path.getsize(file)

    if nbytes > bytelimit:
        print("{} bytes too beaucoup".format(file))
        delete = True
    if nbytes < lowerbytelimit:
        print("{} bytes too few".format(file))
        delete = True

    # Checks for tabs
    with open(file, "r") as f:
        for line in f.readlines():
            if "\t" in line:
                print("{} contains tab".format(file))
                delete=True
                break

    if delete:
        subprocess.run(["rm", "-f", file])
