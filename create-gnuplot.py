#!/usr/bin/python
"""
Helps visualizing 2-dimensional matricies by creating neighbours.out.[index1,...]
files containing the rows in the input file corresponding to the columns in the
neighbours file at the given indizes.

The output is a gnuplot script that will plot the data.

Example:
    python create-gnuplot.py sequoia-1200.txt neighbours.out 0 1 2 3 | gnuplot -p
"""
import sys

if len(sys.argv) < 3:
    print "Usage: dense.in neighbours.out index1[, index2[, ...]]"
    sys.exit(1)

f = open(sys.argv[2], "r")

neighbours = f.readlines() # yup, don't do this with large files
f.close()
f = open(sys.argv[1], "r")
inputdata = f.readlines()
f.close()

fnames = []
for i in range(3, len(sys.argv)):
    idx = int(sys.argv[i])
    fields = neighbours[idx].split()
    fname = sys.argv[2] + "." + str(idx)
    outf = open(fname, "w")
    outf.write("\n".join([inputdata[int(f)].strip() for f in fields]))
    outf.close()
    fnames.append((idx, fname))

print "set size square"
print "set title \"neighbours of object by object ID\""
print "set key font \",7\" spacing .5"
print "plot \"" + sys.argv[1] + "\""
for f in fnames:
    print "replot \"" + f[1] + "\" title \"" + str(f[0]) + "\"" 
