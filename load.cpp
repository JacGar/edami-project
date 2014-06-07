#define _GNU_SOURCE
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/vector.hpp>
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

void usage(const char * argv0) {
  cerr << "Usage: " << argv0 << " <infile>" << endl;
}


/* file import {{{ */
void load_fiile(const string& infilename, matrix_t& out) {
  
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

  int row = 0;
  while ((read = getline(&line, &len, fp)) != -1)  {
    char* next_tk = strtok(line, " ");

    while (next_tk) {
      int idx = atoi(next_tk);
      next_tk = strtok(NULL, " ");
      int val = atoi(next_tk);
      out(row, idx-1) = val;
      next_tk = strtok(NULL, " ");
    }

    ++row;
  }
}
/* }}} */
/* distance implementations {{{ */

typedef double (*distance_fun)(matrix_t &, long unsigned int, long unsigned int);

double manhattan_distance_proxied(matrix_t &m, long unsigned int row1, long unsigned int row2) {
    // 45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    double dist = 0;
    for (unsigned int i = 0; i < m.size2(); ++i) {
        dist += fabs(row1_proxy(i) - row2_proxy(i));
    }
    return dist;
}

double manhattan_distance_builtin(matrix_t &m, long unsigned int row1, long unsigned int row2) {
    // 0.45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    double dist = 0;
    return norm_1(row1_proxy - row2_proxy);
}

double eucledian_distance_builtin(matrix_t &m, long unsigned int row1, long unsigned int row2) {
    // 0.45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    double dist = 0;
    return norm_2(row1_proxy - row2_proxy);
}

double manhattan_distance_manual(matrix_t &m, long unsigned int row1, long unsigned int row2) {
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

double manhattan_distance_manual_cached(matrix_t &m, long unsigned int row1, long unsigned int row2) {
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
      // dist += fabs(*it1_ - *it2_);
      // Euclidean:
      double dist_t = *it1_ - *it2_;
      dist += dist_t*dist_t;
      it2_++; it1_++;
    }
  }
out:
  // manhattan:
  //return dist;
  // eucledian:
  return sqrt(dist);
}

/*}}}*/

template<distance_fun distFun>
void create_distance_matrix(matrix_t &data, const distvector_t &reference_distances_unsorted, double eps, distancematrix_t &dst) {
  size_t ds1 = data.size1();
  for (size_t i = 0; i < ds1; ++i) {
    cout << i << endl;
    for (size_t j = 0; j < i; ++j) {
      if (fabs(reference_distances_unsorted[i].first - reference_distances_unsorted[j].first) <= eps) {
        dst(i,j) = distFun(data, i, j);
      }
    }
    dst(i,i) = 0;
  }
}

double get_distance(const distancematrix_t &dst, size_t a, size_t b) {
  return dst(a,b);
}

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

struct node_meta {
  node_meta() : cluster(UINT_MAX), visited(false), noise(false) { }
  unsigned int cluster;
  bool visited;
  bool noise;
};

/* eps neighbourhood {{{ */
/**
 * get the neighbours for a given distvector index within range eps, including the point itself.
 * Returned are the distvector indices.
 * TODO use list instead of vector (no random access required, but append)
 */
void getneighbours(const distvector_t &reference_distances, const distancematrix_t &dsm, std::vector<size_t>& ret, size_t distvector_idx, double eps) {
  size_t lbound, rbound;
  lbound = rbound = distvector_idx;
  double mid_value = reference_distances[distvector_idx].first;
  ret.push_back(reference_distances[distvector_idx].second);

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
  int current_cluster = 0;
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
    getneighbours(reference_distances, dsm, neighbours, (size_t)i, 2.0);


    if (neighbours.size() < minpts) {
      nodeinfo[obj_idx].noise = true;
      continue;
    }
    
    nodeinfo[obj_idx].cluster = current_cluster++;
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

int main(int argc, char ** argv) {
  string infile;
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  double eps = 0.6;
  size_t minpts = 10;
  bool use_triangle_inequality = true;

  trace("load file");
  infile= argv[1];
  matrix_t data;
  try {
    load_fiile(infile, data);
  } catch (const exception& c) {
    cout << "Error: " << c.what() << endl;
  }

  // basepoints for triangle inequality
  distvector_t reference_distances;
  reference_distances.resize(data.size1());
  
  if (use_triangle_inequality) {
    trace("create reference distances");
    for (unsigned int i = 0; i < data.size1(); ++i) {
      reference_distances[i] = make_pair(eucledian_distance_builtin(data, 0, i), i);
    }
  }
  
  trace("create distancematrix");
  distancematrix_t distancematrix(data.size1(), data.size1());
  create_distance_matrix<eucledian_distance_builtin>(data, reference_distances, eps, distancematrix);

  
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
    fprintf(f, "%d%s\n", metadata[i].cluster, metadata[i].noise?" (noise)":"");
  }
  fclose(f);

  dump_trace();
}
