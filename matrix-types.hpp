#ifndef HAVE_MATRIX_TYPES_HPP
#define HAVE_MATRIX_TYPES_HPP
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
#include "types.hpp"

#ifdef DEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif

using namespace std;


void load_file_clusters(const std::string filename, std::vector<node_meta>& nodeinfo);

template <class T>
class simplematrix {
  /** boost::ublas and armadillo are both not that very good when it comes to supporting
   * sparse matrices. This class is a very, very simple wrapper around basic stl types to
   * provide a sparse matrix.
   *
   * the typical use-case is
   *  - one-time initialization
   *  - calculating distances between row vectors
   */
public:
  typedef vector<pair<size_t, T> > row_type;
  typedef std::vector<std::vector<std::pair<size_t, T> > > contents_type;
  contents_type contents;
  size_t n_cols;
  size_t n_rows;

  simplematrix() : contents(0), n_cols(0), n_rows(0) {

  };

  simplematrix(size_t nrows, size_t ncols) :  contents(nrows), n_cols(ncols), n_rows(nrows) {
  }

  void resize(size_t nrows, size_t ncols) {
    contents.resize(nrows);
    n_rows = nrows;
    n_cols = ncols;
  }

  void row_check(size_t row) const {
    if (row > contents.size()) {
      throw std::runtime_error("bad row: " + boost::lexical_cast<string>(row) + " is greater than the maximum number of rows (" + boost::lexical_cast<string>(contents.size()) +")");
    }
  }

  void col_check(size_t col) const {
    if (col > n_cols) {
      throw std::runtime_error("bad column: " + boost::lexical_cast<string>(col) + " is greater than max cols "
          + boost::lexical_cast<string>(n_cols));
    }
  }

  void set_row(size_t row, const row_type &r) {
    row_check(row);
    contents[row] = r;
  }

  T operator()(size_t row, size_t col) const {
    row_check(row);
    col_check(col);
    row_type &rowvec = contents[row];
    size_t l = 0, r = rowvec.size()-1;
    while (l != r) {
      size_t m = (l+r)/2;
      if (rowvec[m].first < col) {
        l = m;
      } else if (rowvec[m].first > col) {
        r = m;
      } else {
        return rowvec[m];
      }
    }
    return 0;
  }

  typename row_type::const_iterator row_begin(size_t row) const {
    row_check(row);
    return contents[row].begin();
  }

  typename row_type::const_iterator row_end(size_t row) const {
    row_check(row);
    return contents[row].end();
  }
};

template<int norm>
struct mt_sparse_simple {
  typedef simplematrix<double> type;

  static void load_file(const string& infilename, type& out, featurescale_t &featurescale);

  static double calculate_distance(const type &m, size_t row1, const type &m2, size_t row2, const featurescale_t& /*featurescale*/) 
  { 
    typename type::row_type::const_iterator r1 = m.row_begin(row1);
    typename type::row_type::const_iterator r2 = m2.row_begin(row2);

    double sum = 0;
    while (r1 != m.row_end(row1) && r2 != m2.row_end(row2)) {
      if (r1->first < r2->first) {
        r1++;
        continue;
      }
      if (r2->first < r1->first) {
        r2++;
      }
      if (norm == 1) {
        sum += fabs(r1->second - r2->second);
      } else { 
        double tmp = r1->second - r2->second;
        sum += (tmp*tmp);
      }
      r1++; r2++;
    }

    if (norm == 1) {
      return sum;
    } else {
      return sqrt(sum);
    }
  }
};
/* mt_sparse {{{*/
template <int norm>
struct mt_sparse {
  typedef typename arma::SpMat<double> type;

  static void load_file(const string& infilename, type& out, featurescale_t &featurescale);

  static double calculate_distance(const type &m, size_t row1, const type &m2, size_t row2, const featurescale_t& /*featurescale*/) {
    // Note: the following doesn't work (incorrect results) apparently:
    //arma::SpMat<datatype> z = m.row(row1);
    //z -= m2.row(row2);
    //return arma::norm(z, 1);
    // and the following is really slow. Hence mt_sparse_simple.
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
  static void load_file(const string& infilename, type& out, featurescale_t &featurescale);
};


extern template struct mt_dense<1>;
extern template struct mt_dense<2>;
extern template struct mt_sparse<1>;
extern template struct mt_sparse<2>;
extern template struct mt_sparse_simple<1>;
extern template struct mt_sparse_simple<2>;
#endif // HAVE_MATRIX_TYPES_HPP

/* vim: set fdm=marker: */
