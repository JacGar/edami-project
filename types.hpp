#ifndef HAVE_TYPES_HPP
#define HAVE_TYPES_HPP
#include <vector>
#include <map>
#include <string>
#include <stdexcept>

typedef double datatype; // TODO only int supported for now (atoi function calls)
typedef std::vector<datatype> featurescale_t; // holds maximum value. TODO this assumes that all values start with 0

typedef double datatype; // TODO only int supported for now (atoi function calls)
typedef std::vector<std::pair<double, size_t> > distvector_t;
typedef std::vector<datatype> featurescale_t; // holds maximum value. TODO this assumes that all values start with 0

class distancematrix_t : public std::vector<std::vector<double> > {
public:
  distancematrix_t(size_t n, size_t /* ncols*/) {
    resize(n, std::vector<double>(n));
  } 
  double &operator()(size_t row, size_t col) {
    if (row >= size()) {
      throw std::runtime_error("Row index out of bounds");
    }
    if (col >= size()) {
      throw std::runtime_error("Column index out of bounds");
    }
    if (row > col) {
      return (*this)[col][row];
    } else {
      return (*this)[row][col];
    }
  }
  double operator()(size_t row, size_t col) const {
    if (row >= size()) {
      throw std::runtime_error("Row index out of bounds");
    }
    if (col >= size()) {
      throw std::runtime_error("Column index out of bounds");
    }
    if (row > col) {
      return (*this)[col][row];
    } else {
      return (*this)[row][col];
    }
  }
};

struct node_meta {
  node_meta() : cluster(""), visited(false), noise(false) { }
  std::string cluster;
  bool visited;
  bool noise;
  distvector_t neighbours;
};
#endif
