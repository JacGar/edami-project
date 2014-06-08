#!/bin/bash
INPUT=sports
OUTPUT=sports_random
N=1000
echo -n "$N " > $OUTPUT.mat
head -n 1 $INPUT.mat | cut -d' ' -f 2- >> $OUTPUT.mat
tail -n +2 $INPUT.mat > $OUTPUT.mat.tmp.1
paste $INPUT.rclass $OUTPUT.mat.tmp.1 > $OUTPUT.mat.tmp.2
sort -R $OUTPUT.mat.tmp.2 | head -n $N > $OUTPUT.mat.tmp.3
cut -f 1 $OUTPUT.mat.tmp.3 > $OUTPUT.rclass
cut -f 2 $OUTPUT.mat.tmp.3 >> $OUTPUT.mat 
