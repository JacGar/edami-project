#!/bin/bash
# takes a CLUTO sparse matrix and the associated .rclass file, takes N random
# samples out of it and saves it to the OUTPUT file. (This does not update n_elements)
INPUT=sports
OUTPUT=sports_random
N=100
echo -n "$N " > $OUTPUT.mat
head -n 1 $INPUT.mat | cut -d' ' -f 2 | tr -d '\n' >> $OUTPUT.mat
echo -n " " >> $OUTPUT.mat
tail -n +2 $INPUT.mat > $OUTPUT.mat.tmp.1
paste $INPUT.rclass $OUTPUT.mat.tmp.1 > $OUTPUT.mat.tmp.2
sort -R $OUTPUT.mat.tmp.2 | head -n $N > $OUTPUT.mat.tmp.3

cut -f 1 $OUTPUT.mat.tmp.3 > $OUTPUT.rclass
echo $(($(cut -f 2 $OUTPUT.mat.tmp.3 | wc --words | tr -d '\n')/2)) >> $OUTPUT.mat
cut -f 2 $OUTPUT.mat.tmp.3  >> $OUTPUT.mat

