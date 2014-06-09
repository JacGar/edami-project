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

typedef compressed_matrix<int> matrix_t;
typedef symmetric_matrix<double> distancematrix_t; 
typedef std::vector<pair<double, unsigned int> > distvector_t; 
typedef std::vector<double> featurescale_t;

void usage(const char * argv0) {
  cerr << "Usage: " << argv0 << " <infile>" << endl;
}

/* file import {{{ */
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
      featurescale[idx-1] = max(featurescale[idx-1], (double)val);
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

  read = getline(&line, &len, fp);

  if (read == -1) {
    throw runtime_error("Could not read file (empty?)");
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
        featurescale[idx] = max(featurescale[idx], (double)val);
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

double manhattan_distance_manual_cached(matrix_t &m, long unsigned int row1, long unsigned int row2, const featurescale_t& featurescale) {
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
      // Manhattan:
       dist += fabs(*it1_ - *it2_)/featurescale[it1_.index2()];
      // Euclidean:
      // double dist_t = (*it1_ - *it2_)/featurescale[it1_.index2()];
      //dist += dist_t*dist_t;
      it2_++; it1_++;
    }
  }
out:
  // manhattan:
  return dist;
  // eucledian:
  //return sqrt(dist);
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
struct node_meta {
  node_meta() : cluster(UINT_MAX), visited(false), noise(false) { }
  unsigned int cluster;
  bool visited;
  bool noise;
};

/* eps neighbourhood {{{ */
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

void epsneighbourhood(const distvector_t &reference_distances, const distancematrix_t &dsm, double eps, size_t minpts, std::vector<node_meta>& nodeinfo) {
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
    
    nodeinfo[obj_idx].cluster = current_cluster;
    size_t current_cluster_size = 1;
    cout << "Expanding cluster " << current_cluster << " with " << neighbours.size() << " neighbours:" << endl;
    for (size_t j = 0; j < reference_distances.size(); ++j) {
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
      if (nodeinfo[nb_obj_idx].cluster == UINT_MAX) {
        nodeinfo[nb_obj_idx].cluster = current_cluster;
        current_cluster_size++;
      }
    }
    cout << "  Cluster size: " << current_cluster_size << endl;
  }
}

/*}}}*/
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

  string filetype = vm["input-type"].as<string>();

  string infile = vm["input-file"].as<string>();
  double eps = vm["epsilon"].as<double>();
  size_t minpts = vm["minpts"].as<unsigned int>();

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
      reference_distances[i] = make_pair(manhattan_distance_manual_cached(data, 0, i, featurescale), i);
      cout << reference_distances[i].first << " ";
    }
  }
  
  trace("create distancematrix");
  distancematrix_t distancematrix(data.size1(), data.size1());
  create_distance_matrix<manhattan_distance_manual_cached>(data, reference_distances, eps, featurescale, distancematrix);

  
  trace("sort reference distances");
  if (use_triangle_inequality) {
    sort(reference_distances.begin(), reference_distances.end());
  }

  trace("clustering");
  std::vector<node_meta> metadata(data.size1());
  epsneighbourhood(reference_distances, distancematrix, eps, minpts, metadata);

  trace("output");
  FILE *f = fopen("cluster.out", "w");
  for (size_t i = 0; i < metadata.size(); ++i) {
    fprintf(f, "%u%s\n", metadata[i].cluster, metadata[i].noise?" (noise)":"");
  }
  fclose(f);

  dump_trace();
}

/* vim: set fdm=marker: */
