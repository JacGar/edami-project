#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/lexical_cast.hpp>
#include <armadillo>
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
#include "readline.hpp"

#ifdef DEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif

using namespace std;

typedef double datatype; // TODO only int supported for now (atoi function calls)
typedef std::vector<datatype> featurescale_t; // holds maximum value. TODO this assumes that all values start with 0

/* mt_sparse {{{*/
template <int norm>
struct mt_sparse {
  typedef typename arma::SpMat<double> type;

  static void load_file(const string& infilename, type& out, featurescale_t &featurescale) {
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
      throw std::runtime_error("Could not read file (empty?)");
    }

    int nrows, ncols, nnzelem;
    char * tk = strtok(line, " ");
    nrows = atoi(tk);
    tk = strtok(NULL, " ");
    ncols = atoi(tk);
    tk = strtok(NULL, " ");
    if (tk == NULL || strlen(tk) <= 1) {
      throw runtime_error("This is a CLUTO dense matrix, but this tool only supports reading sparse atm");
    }

    nnzelem = atoi(tk);

    arma::umat locations(nnzelem, 2);
    arma::vec values(nnzelem);
    
    featurescale.resize(ncols, 1.0);
    // TODO use batch insert format
    int row = 0;
    while ((read = getline(&line, &len, fp)) != -1)  {
      char* next_tk = strtok(line, " ");

      while (next_tk) {
        int idx = atoi(next_tk);
        next_tk = strtok(NULL, " ");
        int val = atoi(next_tk);
        //out(row, idx-1) = val;
        locations << row << idx-1 << arma::endr;
        values << (double)val;
        featurescale[idx-1] = max((double)featurescale[idx-1], (double)val);
        next_tk = strtok(NULL, " ");
      }
      ++row;
    }
    out = type(locations.t(), values);
  }

  static double calculate_distance(const type &m, size_t row1, const type &m2, size_t row2, const featurescale_t& /*featurescale*/) {
    //arma::SpMat<datatype> z = m.row(row1);
    //z -= m2.row(row2);
    //return arma::norm(z, 1);
    typename type::const_row_iterator r1 = m.begin_row(row1);
    typename type::const_row_iterator r2 = m2.begin_row(row2);

    double sum = 0;
    while (r1 != m.end_row(row1) && r2 != m2.end_row(row2)) {
      if (r1.col() < r2.col()) {
        r1++;
        continue;
      }
      if (r2.col() < r1.col()) {
        r2++;
      }
      if (norm == 1) {
        sum += fabs(*r1 - *r2);
      } else { 
        double tmp = *r1 - *r2;
        sum += (tmp*tmp);
      }
    }

    if (norm == 1) {
      return sum;
    } else {
      return sqrt(sum);
    }
  }
};
/**}}}*/
/* mt_dense {{{*/
template <int norm>
struct mt_dense {
  typedef typename arma::Mat<double> type;

  static double calculate_distance(const type &m, size_t row1, const type &m2, size_t row2, const featurescale_t& /*featurescale*/) {
    return arma::norm(m.row(row1) - m2.row(row2), norm);
  }

  static void load_file(const string& infilename, type& out, featurescale_t &featurescale) {
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
    std::vector<std::vector<datatype> > data;

    while ((read = getline(&line, &len, fp)) != -1)  {
      char* next_tk = strtok(line, " ");
      unsigned int idx = 0;
      std::vector<datatype> inner;
      while (next_tk) {
        datatype val = atoi(next_tk);
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
    out = type(row, data[0].size());

    for (unsigned int i = 0; i < data.size(); ++i) {
      for (unsigned int j = 0; j < data[i].size(); ++j) {
        out(i,j) = data[i][j]; // how very efficient! :/
      }
    }
  }
};


extern template struct mt_dense<1>;
extern template struct mt_dense<2>;
extern template struct mt_sparse<1>;
extern template struct mt_sparse<2>;
/* vim: set fdm=marker: */
