#include "matrix-types.hpp"
#include "readline.hpp"

template <int norm>
/*static */void mt_sparse_simple<norm>::load_file(const string& infilename, type& out, featurescale_t &featurescale) {
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

  int nrows, ncols;
  char * tk = strtok(line, " ");
  nrows = atoi(tk);
  tk = strtok(NULL, " ");
  ncols = atoi(tk);
  tk = strtok(NULL, " ");
  if (tk == NULL || strlen(tk) <= 1) {
    throw runtime_error("This is a CLUTO dense matrix, but this tool only supports reading sparse atm");
  }

  /* nnzelem = atoi(tk); */

  out = type(nrows, ncols);
  
  featurescale.resize(ncols, 1.0);
  // TODO use batch insert format
  int row = 0;
  while ((read = getline(&line, &len, fp)) != -1)  {
    type::row_type new_row;
    char* next_tk = strtok(line, " ");

    while (next_tk) {
      int idx = atoi(next_tk);
      next_tk = strtok(NULL, " ");
      int val = atoi(next_tk);
      new_row.push_back(make_pair(idx-1, val));
      featurescale[idx-1] = max((double)featurescale[idx-1], (double)val);
      next_tk = strtok(NULL, " ");
    }
    out.set_row(row, new_row);
    ++row;
  }
}

#ifdef USE_ARMADILLO
template <int norm>
/* static */void mt_sparse<norm>::load_file(const string& infilename, type& out, featurescale_t &featurescale) {
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
  if (row != nrows) {
    // don't throw an exception here, someone might just have wanted a smaller input file
    // and trimmed it ;)
    cerr << endl << "WARNING: number of expected rows does not match actual rows in " <<
      infilename << ", file might be damaged" << endl;
  }
  out = type(locations.t(), values);
}
#endif // USE_ARMADILLO

template<int norm>
/*static */ void mt_dense_simple<norm>::load_file(const string& infilename, type& out, featurescale_t &featurescale) {
  FILE *fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  fp = fopen(infilename.c_str(), "r");
  if (fp == NULL) {
    throw runtime_error("Could not open file " + infilename);
  }

  unsigned int row = 0;

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

    if (out.size() && inner.size() != out[0].size()) {
      throw runtime_error("matrix dimension mismatch on row " + boost::lexical_cast<string>(row+1));
    }
    ++row;
    out.push_back(inner);
  }

  out.update();
  if (!out.size()) {
    throw runtime_error("Input file didn't contain anything apparently");
  }
}

#ifdef USE_ARMADILLO
template<int norm>
/*static */ void mt_dense<norm>::load_file(const string& infilename, type& out, featurescale_t &featurescale) {
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
#endif

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

#ifdef USE_ARMADILLO
template struct mt_dense<1>;
template struct mt_dense<2>;
template struct mt_sparse<1>;
template struct mt_sparse<2>;
#endif
template struct mt_sparse_simple<1>;
template struct mt_sparse_simple<2>;
template struct mt_dense_simple<1>;
template struct mt_dense_simple<2>;
