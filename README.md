# Clusterer

This tool has been created initially as a part of the "EDAMI" Data Mining class
at Politechnika Warszawska, but has been extended to allow for a couple more
features. The main features of the program are:

* Dense and sparse (CLUTO file format) dataset support
* EPS- and K-Nearest Neighbours tests
* Clustering using DBSCAN
* Supports both manhattan (1-) and euclidean (2-) norm
* Makes use of triangle inequality for performance, can optionally be disabled
* Uses boost for program options, no other external dependencies
  * Support for armadillo matrices can be enabled at compile time

EPS- and K-Nearest-Neighbour tests will dump the neighbour list to a file
(can be disabled using `--test-out ""`).
If clustering is performed (or cluster data is available using
`--cluster-method=file`) a membership file will be generated, which will
assign each node a cluster based on the most frequent neighbour nodes
in the eps- or k-neighbourhood.
