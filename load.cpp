#define _GNU_SOURCE
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <ctime>

using namespace std;
using namespace boost::numeric::ublas;

typedef mapped_matrix<int> matrix_t;

void usage(const char * argv0) {
  cerr << "Usage: " << argv0 << " <infile>" << endl;
}

void load_fiile(const string& infilename, matrix_t& out) {
  
  FILE *fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(infilename.c_str(), "r");
  if (fp == NULL) {
    throw runtime_error("Could not open file " + infilename);
  }

  char buffer[1024];
  
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

const int ntrc = 3;
const char* trace_points[ntrc] = {"start", "read", "dump"};
double trace_tstamps[ntrc];

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

double manhattan_distance_builtinn1(matrix_t &m, long unsigned int row1, long unsigned int row2) {
    // 0.45 seconds for n rows
    boost::numeric::ublas::matrix_row<matrix_t> row1_proxy(m, row1);
    boost::numeric::ublas::matrix_row<matrix_t> row2_proxy(m, row2);
    double dist = 0;
    return norm_1(row1_proxy - row2_proxy);
}

double manhattan_distance_manual(matrix_t &m, long unsigned int row1, long unsigned int row2) {
  // about 45 seconds for n rows. Adding const does not yield any improvement.
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
  // for the brave: changing anything about the matrix m might yield an invalid iterators, and crashes.
  // this is a sort of workaround fr the fact that we cannot access rows directly, by caching
  // the iterators. This, of course, assumes that the matrix m never changes, either it's values or
  // its position in the memory.
  static matrix_t::const_iterator1 *it1_c = NULL, *it2_c = NULL;
  static long unsigned int pos1, pos2;
  if (it1_c == NULL || it2_c == NULL) {
      *it1_c = m.begin1();
      *it2_c = m.begin1();
      pos1 = pos2 = 0;
  }

  matrix_t::const_iterator1 &it1 = *it1_c;
  matrix_t::const_iterator1 &it2 = *it2_c;
  while (pos1 > row1) { it1--; pos1--; }
  while (pos1 < row1) { it1++; pos1++; }
  while (pos2 > row2) { it2--; pos2--; }
  while (pos2 < row2) { it2++; pos2++; }

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

void trace() {
  static int current = 0;
  if (current < ntrc) {
    trace_tstamps[current++] = clock();
  }
}

void dump_trace() {
  for (int i = 1; i < ntrc; ++i) {
    cout << trace_points[i] << "\t" << (trace_tstamps[i]-trace_tstamps[i-1])/(double)CLOCKS_PER_SEC << endl;
  }
}

int main(int argc, char ** argv) {
  string infile;
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }

  trace();
  infile= argv[1];
  matrix_t data;
  try {
    load_fiile(infile, data);
  } catch (const exception& c) {
    cout << "Error: " << c.what() << endl;
  }

  trace();

  // print the manhattan distance from the first element to every other
  for (unsigned int i = 0; i < data.size1(); ++i) {
    cout << manhattan_distance_manual(data, 0, i) << " ";
    cout.flush();
  }
  cout << endl;

  trace();
  dump_trace();
}
