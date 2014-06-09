/* includes and definitions {{{*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <ctime>
#include <cfloat>
#include <algorithm>
#include <map> // pair

using namespace std;
using namespace boost::numeric::ublas;

typedef int datatype; // TODO only int supported for now (atoi function calls)
typedef compressed_matrix<datatype> matrix_t;
typedef symmetric_matrix<double> distancematrix_t;
typedef std::vector<pair<double, unsigned int> > distvector_t;
typedef std::vector<datatype> featurescale_t; // holds maximum value. TODO this assumes that all values start with 0

struct node_meta {
  node_meta() : cluster(""), visited(false), noise(false) { }
  string cluster;
  bool visited;
  bool noise;
};

/* }}} */

/* file import {{{ */

#ifdef _WIN32
/* This code is public domain -- Will Hartung 4/9/09 */
size_t getline(char **lineptr, size_t *n, FILE *stream) {
    char *bufptr = NULL;
    char *p = bufptr;
    size_t size;
    int c;

    if (lineptr == NULL) {
    	return -1;
    }
    if (stream == NULL) {
    	return -1;
    }
    if (n == NULL) {
    	return -1;
    }
    bufptr = *lineptr;
    size = *n;

    c = fgetc(stream);
    if (c == EOF) {
    	return -1;
    }
    if (bufptr == NULL) {
    	bufptr = (char*)malloc(128);
    	if (bufptr == NULL) {
    		return -1;
    	}
    	size = 128;
    }
    p = bufptr;
    while(c != EOF) {
    	if ((p - bufptr) > (size - 1)) {
    		size = size + 128;
    		bufptr = (char*)realloc(bufptr, size);
    		if (bufptr == NULL) {
    			return -1;
    		}
    	}
    	*p++ = c;
    	if (c == '\n') {
    		break;
    	}
    	c = fgetc(stream);
    }

    *p++ = '\0';
    *lineptr = bufptr;
    *n = size;

    return p - bufptr - 1;
}
#endif

void load_file_cluto(const string& infilename, matrix_t& out, featurescale_t &featurescale) {
  FILE *fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(infilename.c_str(), "r");
  if (fp == NULL) {
    throw runtime_error("Could not open file " + infilename);
  }

  read = getline(&line, &len, fp);

  if (read == -1) {
    throw runtime_error("Could not read file (empty?)");
  }

  int nrows, ncols, nnzelem;
  char * tk = strtok(line, " ");
  nrows = atoi(tk);
  tk = strtok(NULL, " ");
  ncols = atoi(tk);
  tk = strtok(NULL, " ");
  if (strlen(tk) <= 1) {
    throw runtime_error("This is a CLUTO dense matrix, but this tool only supports reading sparse atm");
  }

  nnzelem = atoi(tk);

  out = matrix_t(nrows, ncols, nnzelem);
  featurescale.resize(ncols, 1.0);

  int row = 0;
  while ((read = getline(&line, &len, fp)) != -1)  {
    char* next_tk = strtok(line, " ");

    while (next_tk) {
      int idx = atoi(next_tk);
      next_tk = strtok(NULL, " ");
      int val = atoi(next_tk);
      out(row, idx-1) = val;
      featurescale[idx-1] = max((double)featurescale[idx-1], (double)val);
      next_tk = strtok(NULL, " ");
    }
    ++row;
  }
}


template<typename dtype>
void load_file_plain(const string& infilename, matrix_t& out, featurescale_t &featurescale) {
  // this function reads a dense matrix into a sparse filetype, until i can figure out if there
  // is a more efficient way to do this. Note that this will be pretty slow for larger datasets,
  // both using the sparse datatype and the actual "read" implementation used here.
  FILE *fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(infilename.c_str(), "r");
  if (fp == NULL) {
    throw runtime_error("Could not open file " + infilename);
  }

  unsigned int row = 0;
  std::vector<std::vector<dtype> > data;

  while ((read = getline(&line, &len, fp)) != -1)  {
    char* next_tk = strtok(line, " ");
    unsigned int idx = 0;
    std::vector<dtype> inner;
    while (next_tk) {
      dtype val = atoi(next_tk);
      inner.push_back(val);

      if (featurescale.size() <= idx) {
        featurescale.push_back(val);
      } else {
        featurescale[idx] = max((double)featurescale[idx], (double)val);
      }
      idx++;
      next_tk = strtok(NULL, " ");
    }

    if (data.size() && inner.size() != data[0].size()) {
      throw runtime_error("matrix dimension mismatch on row " + boost::lexical_cast<string>(row+1));
    }
    ++row;
    data.push_back(inner);
  }

  if (!data.size()) {
    throw runtime_error("Input file didn't contain anything apparently");
  }
  out = matrix_t(row, data[0].size(), row*data[0].size());

  for (unsigned int i = 0; i < data.size(); ++i) {
    for (unsigned int j = 0; j < data[i].size(); ++j) {
      out(i,j) = data[i][j]; // how very efficient! :/
    }
  }
}

void load_file_clusters(const std::string filename, std::vector<node_meta>& nodeinfo) {
  FILE *fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(filename.c_str(), "r");
  if (fp == NULL) {
    throw runtime_error("Could not open cluster label file " + filename);
  }

  unsigned int row = 0;

  while ((read = getline(&line, &len, fp)) != -1)  {
    // remove trailing whitespace
    size_t offset = strlen(line);
    while (offset-- > 0) {
      if (isspace(line[offset])) {
        line[offset] = '\0';
      }
    }
    if (!strlen(line)) {
      throw runtime_error("Empty line in cluster label file: #" + boost::lexical_cast<string>(row));
    }
    if (nodeinfo.size() <= row) {
      throw runtime_error("Too many rows in cluster label file");
    }
    nodeinfo[row].cluster = string(line);
    row++;
  }
  if (row != nodeinfo.size()) {
    throw runtime_error("Too few lines in cluster label file");
  }
}

/* }}} */
/* distance implementations {{{ */

typedef double (*distance_fun)(matrix_t &, long unsigned int, long unsigned int, const featurescale_t&);

double manhattan_distance_proxied(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t& featurescale) {
    // 45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    double dist = 0;
    for (unsigned int i = 0; i < m.size2(); ++i) {
        dist += fabs(row1_proxy(i) - row2_proxy(i));
    }
    return dist;
}

double manhattan_distance_builtin(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t &featurescale) {
    // 0.45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    return norm_1(row1_proxy - row2_proxy);
}

double eucledian_distance_builtin(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t &featurescale) {
    // 0.45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    return norm_2(row1_proxy - row2_proxy);
}

double manhattan_distance_manual(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t& featurescale) {
  // about 18 seconds for n rows. Adding const does not yield any improvement.
  // is there a way to access rows directly, not with this fucking magic thing?
  matrix_t::const_iterator1 it1 = m.begin1();
  while (row1-- > 0) {
    it1++;
  }

  matrix_t::const_iterator1 it2 = m.begin1();
  while (row2-- > 0) {
    it2++;
  }

  double dist = 0;
  matrix_t::const_iterator2 it1_ = it1.begin();
  matrix_t::const_iterator2 it2_ = it2.begin();
  while (it1_ != it1.end() && it2_ != it2.end()) {
    while (it1_.index2() < it2_.index2()) {
      it1_++;
      if (it1_ == it1.end())
          goto out;
    }
    while (it2_.index2() < it1_.index2()) {
      it2_++;
      if (it2_ == it2.end())
          goto out;
    }
    if (it1_.index2() == it2_.index2() && it1_ != it1.end() && it2_ != it2.end()) {
      dist += fabs(*it1_ - *it2_);
      it2_++; it1_++;
    }
  }
out:
  return dist;
}

template<class T>
struct c_euclid {
  static inline double aggregate(T a, T b, double fscl) {
    double dist = (a-b)/fscl;
    return dist*dist;
  }
  static inline double finalize(double dist) {
    return sqrt(dist);
  }
};

template<class T>
struct c_manhattan {
  static inline double aggregate(T a, T b, double fscl) {
    return fabs(a-b)/fscl;
  }
  static inline double finalize(double dist) {
    return dist;
  }
};

template<class aggr>
double distance_manual_cached(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t& featurescale) {
  // this takes about 0.58 sec
  // for the brave: changing anything about the matrix m might yield an invalid iterators, and crashes.
  // this is a sort of workaround fr the fact that we cannot access rows directly, by caching
  // the iterators. This, of course, assumes that the matrix m never changes, either it's values or
  // its position in the memory.
  static matrix_t::const_iterator1 *it1_c = NULL, *it2_c = NULL;
  static long unsigned int pos1, pos2;
  if (it1_c == NULL || it2_c == NULL) {
    it1_c = new matrix_t::const_iterator1(m.begin1());
    it2_c = new matrix_t::const_iterator1(m.begin1());
    pos1 = pos2 = 0;
  }
  matrix_t::const_iterator1 &it1 = *it1_c;
  matrix_t::const_iterator1 &it2 = *it2_c;

  // move the iterator to the correct position.
  while (pos1 > row1) { it1--; pos1--; }
  while (pos1 < row1) { if (it1 == m.end1()) break; it1++; pos1++; }
  while (pos2 > row2) { it2--; pos2--; }
  while (pos2 < row2) { if (it2 == m.end1()) break; it2++; pos2++; }

  double dist = 0;
  matrix_t::const_iterator2 it1_ = it1.begin();
  matrix_t::const_iterator2 it2_ = it2.begin();
  while (it1_ != it1.end() && it2_ != it2.end()) {
    while (it1_.index2() < it2_.index2()) {
      it1_++;
      if (it1_ == it1.end())
          goto out;
    }
    while (it2_.index2() < it1_.index2()) {
      it2_++;
      if (it2_ == it2.end())
          goto out;
    }
    if (it1_.index2() == it2_.index2() && it1_ != it1.end() && it2_ != it2.end()) {
      dist += aggr::aggregate(*it1_, *it2_, featurescale[it1_.index2()]);
      *it1_++; *it2_++;
    }
  }
out:
  return aggr::finalize(dist);
}

template<class aggr>
double distance_find(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t& featurescale) {

  matrix_t::const_iterator1 it1 = m.find1(1, row1, 0);
  matrix_t::const_iterator1 it2 = m.find1(1, row2, 0);

  double dist = 0;
  matrix_t::const_iterator2 it1_ = it1.begin();
  matrix_t::const_iterator2 it2_ = it2.begin();
  while (it1_ != it1.end() && it2_ != it2.end()) {
    while (it1_.index2() < it2_.index2()) {
      it1_++;
      if (it1_ == it1.end())
          goto out;
    }
    while (it2_.index2() < it1_.index2()) {
      it2_++;
      if (it2_ == it2.end())
          goto out;
    }
    if (it1_.index2() == it2_.index2() && it1_ != it1.end() && it2_ != it2.end()) {
      dist += aggr::aggregate(*it1_, *it2_, featurescale[it1_.index2()]);
      //cout << "aggregating (" << it1_.index1() <<", " << it1_.index2() <<") = " << *it1_ << " and "
      //    << "(" << it2_.index1() <<", " << it2_.index2() <<") = " << *it2_ << endl;
      *it1_++; *it2_++;
    }
  }
out:
  //cout << "finalizing " << dist << " " << aggr::finalize(dist) << endl;
  return aggr::finalize(dist);
}

template<class aggr>
double distance_find_2(matrix_t &m1, long unsigned int row1, matrix_t &m2, long unsigned int row2, const featurescale_t& featurescale) {

  matrix_t::const_iterator1 it1 = m1.find1(1, row1, 0);
  matrix_t::const_iterator1 it2 = m2.find1(1, row2, 0);

  double dist = 0;
  matrix_t::const_iterator2 it1_ = it1.begin();
  matrix_t::const_iterator2 it2_ = it2.begin();
  while (it1_ != it1.end() && it2_ != it2.end()) {
    while (it1_.index2() < it2_.index2()) {
      it1_++;
      if (it1_ == it1.end())
          goto out;
    }
    while (it2_.index2() < it1_.index2()) {
      it2_++;
      if (it2_ == it2.end())
          goto out;
    }
    if (it1_.index2() == it2_.index2() && it1_ != it1.end() && it2_ != it2.end()) {
      dist += aggr::aggregate(*it1_, *it2_, featurescale[it1_.index2()]);
      //cout << "aggregating (" << it1_.index1() <<", " << it1_.index2() <<") = " << *it1_ << " and "
      //    << "(" << it2_.index1() <<", " << it2_.index2() <<") = " << *it2_ << endl;
      *it1_++; *it2_++;
    }
  }
out:
  //cout << "finalizing " << dist << " " << aggr::finalize(dist) << endl;
  return aggr::finalize(dist);
}

/*}}}*/
/** distance matrix {{{ */
template<distance_fun distFun>
void create_distance_matrix(matrix_t &data, const distvector_t &reference_distances_unsorted, double eps, const featurescale_t& featurescale, distancematrix_t &dst) {
  size_t ds1 = data.size1();
  for (size_t i = 0; i < ds1; ++i) {
    cout << i << endl;
    for (size_t j = 0; j < i; ++j) {
      if (fabs(reference_distances_unsorted[i].first - reference_distances_unsorted[j].first) <= eps) {
        dst(i,j) = distFun(data, i, j, featurescale);
      }
    }
    dst(i,i) = 0;
  }
}

double get_distance(const distancematrix_t &dst, size_t a, size_t b) {
  return dst(a,b);
}
/**}}}*/
/** get_next {{{*/
/**
 * Helper function to walk left/right in a distance array, gradually returning values that are farther away.
 * Initialize this function by setting lbound and rbound to the offset in the vector where the value mid_value
 * can be found. either lbond or rbound will be modified, both should be passed to the next call.
 *
 * returns  the index of the next row and the distance to it, or (UINT_MAX, 0.0) when the end of the array has been reached.
 */
pair<unsigned int, double> get_next(const distvector_t &distvec, const int mid_value, size_t &lbound, size_t &rbound) {
  size_t veclen = distvec.size();
  if (veclen == 0 || ((lbound == 0) && (rbound >= veclen-1))) {
    return make_pair(UINT_MAX, 0.0);
  }

  double dist_l = DBL_MAX, dist_r = DBL_MAX;
  if (lbound > 0) {
    // distvec is sorted ascending, hopefully.
    // TODO sqrt, fabs?
    dist_l = mid_value - distvec[lbound-1].first;
  }

  if (rbound < veclen-1) {
    dist_r = distvec[rbound+1].first - mid_value;
  }

  // either dist_r or dist_l is something else than -1 (we checked the "at bounds end" error
  // at the beginning of the function
  if (/*dist_r == DBL_MAX || */ dist_l <= dist_r) {
    lbound--;
    return make_pair(distvec[lbound].second, dist_l);
  }
  if (/*dist_l == DBL_MAX || */ dist_r <= dist_l) {
    rbound++;
    return make_pair(distvec[rbound].second, dist_r);
  }

  throw runtime_error("obligatory 'this should never happen'");
}
/**}}}*/

/* DBSCAN {{{ */
/**
 * get the neighbours for a given distvector index within range eps, excluding the point itself.
 * Returned are the distvector indices.
 * TODO use list instead of vector (no random access required, but append)
 */
void getneighbours(const distvector_t &reference_distances, const distancematrix_t &dsm, std::vector<size_t>& ret, size_t distvector_idx, double eps) {
  size_t lbound, rbound;
  lbound = rbound = distvector_idx;
  double mid_value = reference_distances[distvector_idx].first;

  pair<unsigned int, double> next_index;
  while ((next_index = get_next(reference_distances, mid_value, lbound, rbound)).first != UINT_MAX) {
    if (next_index.second > eps) {
      break;
    }
    // calculate real distance:
    double dist = get_distance(dsm, reference_distances[distvector_idx].second, reference_distances[next_index.first].second);
    if (dist <= eps) {
      ret.push_back(next_index.first);
    }
  }
}

void dbscan(const distvector_t &reference_distances, const distancematrix_t &dsm, double eps, size_t minpts, std::vector<node_meta>& nodeinfo) {
  int current_cluster = -1;
  std::vector<size_t> neighbours;
  for (size_t i = 0; i < reference_distances.size(); ++i) {
    ssize_t obj_idx = reference_distances[i].second;
    // skip visited nodes
    if (nodeinfo[obj_idx].visited) {
      continue;
    }
    nodeinfo[obj_idx].visited = true;

    // get neighbours
    neighbours.clear();
    neighbours.push_back(obj_idx);
    getneighbours(reference_distances, dsm, neighbours, (size_t)i, eps);

    if (neighbours.size() < minpts) {
      nodeinfo[obj_idx].noise = true;
      continue;
    }

    current_cluster++;
    string current_cluster_s = boost::lexical_cast<string>(current_cluster);

    nodeinfo[obj_idx].cluster = current_cluster_s;
    size_t current_cluster_size = 1;
    cout << "Expanding cluster " << current_cluster << " with " << neighbours.size() << " neighbours:" << endl;
    for (size_t j = 0; j < neighbours.size(); ++j) {
      size_t nb_obj_idx = reference_distances[neighbours[j]].second;
      if (nodeinfo[nb_obj_idx].visited == false) {
        nodeinfo[nb_obj_idx].visited = true;
        std::vector<size_t> nneighbours;
        getneighbours(reference_distances, dsm, nneighbours, neighbours[j], eps);
        if (nneighbours.size() >= minpts) {
          neighbours.insert(neighbours.end(), nneighbours.begin(), nneighbours.end());
          cout << "  Added " << nneighbours.size() << " new neighbours" << endl;
        }

      }
      if (nodeinfo[nb_obj_idx].cluster == "") {
        nodeinfo[nb_obj_idx].cluster = current_cluster_s;
        current_cluster_size++;
        nodeinfo[nb_obj_idx].noise=false;
      }
    }
    cout << "  Cluster size: " << current_cluster_size << endl;
  }
}

/*}}}*/
/* membership tests {{{*/
template<class aggr>
void test_epsneighbourhood(matrix_t& data, matrix_t &test_data, distvector_t& reference_distances, featurescale_t &featurescale, const std::vector<node_meta> &nodemeta, double eps, std::vector<node_meta> &nodemetaout) {
  std::vector<size_t> result_indizes;
  nodemetaout.resize(test_data.size1());
  for (size_t i = 0; i < test_data.size1(); ++i) {
    for (size_t j = 0; j < data.size1(); ++j) {
      double dist = distance_find_2<aggr>(test_data, i, data, j, featurescale);
      if (dist <= eps) {
        result_indizes.push_back(j);
      }
    }

    map<string, size_t> occurences;
    for (size_t j = 0; j < result_indizes.size(); ++j) {
      string cl = nodemeta[result_indizes[j]].cluster;
      occurences[cl]++;
    }
    if (!occurences.size()) {
      nodemetaout[i].noise = true;
    }
    size_t cur_max = 0;
    string cur_max_lbl;
    for (std::map<string, size_t>::iterator it = occurences.begin(); it != occurences.end(); ++it) {
      if (it->second > cur_max) {
        cur_max_lbl = it->first;
        cur_max = it->second;
      }
    }
    nodemetaout[i].cluster = cur_max_lbl;
  }
} 
/* }}} */
/* trace code {{{ */

std::vector<const char*> trace_points;
std::vector<double> trace_tstamps;

void trace(const char *tp) {
  trace_points.push_back(tp);
  trace_tstamps.push_back(clock());
}

void dump_trace() {
  trace("");
  for (size_t i = 0; i < trace_points.size()-1; ++i) {
    cout << trace_points[i] << "\t" << (trace_tstamps[i+1]-trace_tstamps[i])/(double)CLOCKS_PER_SEC << endl;
  }
}

/* }}} */

namespace po = boost::program_options;

int main(int argc, char ** argv) {

  po::options_description desc("program options");
  desc.add_options()
    ("help,h", "show usage")
    ("epsilon,e", po::value<double>()->default_value(0.5), "epsilon")
    ("minpts,m", po::value<unsigned int>()->default_value(10), "minimum number of points to form a cluster")
    ("no-triangle-inequality,T", "don't use triangle inequality")
    ("scale,s", "use feature scaling")
    ("input-file", "input file")
    ("input-type,t", po::value<string>()->default_value("cluto"), "File type of the input file. Possible values are:\n"
                                                                  "  cluto - CLUTO file format (sparse)\n"
                                                                  "  plain - space seperated features, each in a new line.")
    ("norm,n", po::value<int>()->default_value(1), "norm to use (1 or 2)")
    ("cluster-method,c", po::value<string>()->default_value("dbscan"), "Cluster method. Possible values are\n"
                                                                      "  dbscan - cluster using dbscan\n"
                                                                      "  file   - read labels from file\n")
    ("labels,l", po::value<string>()->default_value("no"), "labels file (required for --cluster-method file)")
    ("test-membership", po::value<string>()->default_value("no"), "test membership method, possible values:\n"
                                            "  no  - don't test membership\n"
                                            "  eps - use eps neighbourhood\n"
                                            "  k   - use k-nearest-neighbours")
    ("test-file", po::value<string>(), "file to read membership test values from")
    ("test-parameter", po::value<double>(), "membership test parameter (eps or k)")
  ;

  po::positional_options_description p;
  p.add("input-file", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 1;
  }

  bool use_triangle_inequality = true;
  bool use_feature_scaling = false;

  if (vm.count("no-triangle-inequality")) {
    use_triangle_inequality = false;
  }

  if (vm.count("scale")) {
    use_feature_scaling = true;
  }

  if (!vm.count("input-file")) {
    cerr << "Error: input file required." << endl;
    cout << desc << endl;
    return 1;
  }

  const int norm = vm["norm"].as<int>();
  if (norm != 1 && norm != 2) {
    cerr << "Only norms 1 or 2 are supported" << endl;
    cerr << desc << endl;
    return 1;
  }

  const string cluster_method = vm["cluster-method"].as<string>();
  string label_file;
  if (cluster_method == "file") {
    if (!vm.count("labels")) {
      cerr << "Error: --cluster-method=file requires a --labels file" << endl;
      cerr << desc << endl;
      return 1;
    }
    label_file = vm["labels"].as<string>();
  } else if (cluster_method != "dbscan") {
    cerr << "Error: invalid value for --cluster-method" << endl;
    cerr << desc << endl;
    return 1;
  }

  string filetype = vm["input-type"].as<string>();

  string infile = vm["input-file"].as<string>();
  double eps = vm["epsilon"].as<double>();
  size_t minpts = vm["minpts"].as<unsigned int>();

  string test_membership = vm["test-membership"].as<string>();
  int test_membership_k = 0;
  double test_membership_eps = 0.0;
  string test_membership_filename = "";

  if (test_membership != "no") {
    if (test_membership == "eps") {
      if (!vm.count("test-parameter")) {
        throw runtime_error("eps membership type requires a --test-parameter");
      }
      test_membership_eps = vm["test-parameter"].as<double>();
    } else if (test_membership == "k") { 
      if (!vm.count("test-parameter")) {
        throw runtime_error("k-nearest-neighbours membership type requires a --test-parameter");
      }
      test_membership_k = (int)vm["test-parameter"].as<double>();
    } else {
      throw runtime_error("unsupported membership test type: " + test_membership);
    }
    if (!vm.count("test-file")) {
      throw runtime_error("membership test requires --test-file");
    }
    test_membership_filename = vm["test-file"].as<string>();
  }

  cout << "reading from " << infile << ", eps=" << eps <<", minpts=" << minpts << ", "
    << (use_triangle_inequality?"":"not") << " using triangle inequality, "
    << (use_feature_scaling?"":"not") << " using feature scaling." << endl;


  trace("load file");
  matrix_t data;
  featurescale_t featurescale;
  try {
    if (filetype == "cluto") {
      load_file_cluto(infile, data, featurescale);
    } else if (filetype == "plain") {
      load_file_plain<unsigned int>(infile, data, featurescale);
    } else {
      cerr << "Error: unknown file type " << filetype << endl;
      cerr << desc << endl;
      return 1;
    }
  } catch (const exception& c) {
    cout << "Error: " << c.what() << endl;
    return 1;
  }

  if (!use_feature_scaling) {
    featurescale.clear();
    featurescale.insert(featurescale.end(), data.size2(), 1.0);
  }

  // basepoints for triangle inequality
  distvector_t reference_distances;
  reference_distances.resize(data.size1());

  if (use_triangle_inequality) {
    trace("create reference distances");
    for (unsigned int i = 0; i < data.size1(); ++i) {
      if (norm == 1) {
        reference_distances[i] = make_pair(distance_find<c_manhattan<int> >(data, 0, i, featurescale), i);
      } else {
        reference_distances[i] = make_pair(distance_find<c_euclid<int> >(data, 0, i, featurescale), i);
      }
      cout << reference_distances[i].first << " ";
    }
    cout << endl;
  }

  trace("create distancematrix");
  distancematrix_t distancematrix(data.size1(), data.size1());
  if (norm == 1) {
    create_distance_matrix<distance_find<c_manhattan<int> > >(data, reference_distances, eps, featurescale, distancematrix);
  } else {
    create_distance_matrix<distance_find<c_euclid<int> > >(data, reference_distances, eps, featurescale, distancematrix);
  }

  trace("sort reference distances");
  if (use_triangle_inequality) {
    sort(reference_distances.begin(), reference_distances.end());
  }

  std::vector<node_meta> metadata(data.size1());
  if (cluster_method == "dbscan") {
    trace("DBSCAN");
    dbscan(reference_distances, distancematrix, eps, minpts, metadata);
  } else if (cluster_method == "file") {
    trace("label-load");
    load_file_clusters(label_file, metadata);
  }

  // Test membership if wanted
  if (test_membership != "no") {
    matrix_t test_data;
    featurescale_t test_featurescale;
    try {
      if (filetype == "cluto") {
        load_file_cluto(test_membership_filename, test_data, test_featurescale);
      } else {
        load_file_plain<datatype>(test_membership_filename, test_data, test_featurescale);
      }
    } catch (const runtime_error& e) {
      cerr << "Error while loading test member ship data: " << e.what() << endl;
      return 1;
    }
    
    if (test_membership == "eps") {
      //test_epsneighbourhood(data, test_data, reference_distances, test_membership_eps); 
    }
  }

  trace("output");
  FILE *f = fopen("cluster.out", "w");
  for (size_t i = 0; i < metadata.size(); ++i) {
    fprintf(f, "%s%s\n", metadata[i].cluster.c_str(), metadata[i].noise?" (noise)":"");
  }
  fclose(f);

  dump_trace();
}

/* vim: set fdm=marker: */
