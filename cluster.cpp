/* includes and definitions {{{*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/program_options.hpp>
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
#include "matrix-types.hpp"

#ifdef DEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif

#ifndef _WIN32
// not sure how this behaves with windows, so better disable it for now.
#define SHOW_PROGRESS_UPDATE
#endif
 
using namespace std;

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
    printf("%-30s %3.4f\n", trace_points[i], (trace_tstamps[i+1]-trace_tstamps[i])/(double)CLOCKS_PER_SEC);
  }
}

#ifdef SHOW_PROGRESS_UPDATE
void progress(const string& msg, size_t counter, size_t total = 0) {
  static string last_progress;
  if (counter % 50 != 0) {
    return;
  }
  if (string(msg) != last_progress) {
    if (last_progress.size()) {
      printf("%20s done.                                                        \n", last_progress.c_str());
    }
    last_progress = msg;
  } else {
    printf("%20s: %5.1f%% (%lu of %lu)\r", msg.c_str(), (total==0)?100.0:100.0*counter/total, counter, total);
    cout.flush();
  }
}
#else
void progress(const string &msg, size_t counter, size_t total = 0) {
  static string last_progress;
  if (counter % 500 != 0) {
    return;
  }
  if (string(msg) != last_progress) {
    if (last_progress.size()) {
      printf("%20s done.\n", last_progress.c_str());
    }
    last_progress = msg;
  } else {
    printf("%20s: %5.1f%% (%lu of %lu)\n", msg.c_str(), (total==0)?100.0:100.0*counter/total, counter, total);
    cout.flush();
  }
}
#endif

/* }}} */
/* file import {{{ */


/* }}} */
/* distance matrix {{{ */
template<class mt>
void create_distance_matrix(typename mt::type &data, const distvector_t &reference_distances_unsorted, double eps, const featurescale_t& featurescale, distancematrix_t &dst) {
  size_t ds1 = data.n_rows;
  for (size_t i = 0; i < ds1; ++i) {
    progress("creating distance matrix", i, ds1);
    for (size_t j = 0; j < i; ++j) {
      if (fabs(reference_distances_unsorted[i].first - reference_distances_unsorted[j].first) <= eps) {
        dst(i,j) = mt::calculate_distance(data, i, data, j, featurescale);
      }
    }
    dst(i,i) = 0;
  }
}

double get_distance(const distancematrix_t &dst, size_t a, size_t b) {
  double d = dst(a,b);
  //if (d == 0 && a != b) {
  //  throw runtime_error("got 0 distance!");
  //}
  return d;
}
/**}}}*/

/* get_next {{{*/
/**
 * Helper function to walk left/right in a distance array, gradually returning values that are farther away.
 * Initialize this function by setting lbound and rbound to the offset in the vector where the value mid_value
 * can be found. either lbond or rbound will be modified, both should be passed to the next call.
 *
 * returns  the index of the next row and the distance to it, or (UINT_MAX, 0.0) when the end of the array has been reached.
 */
pair<unsigned int, double> get_next(const distvector_t &distvec, const double mid_value, size_t &lbound, size_t &rbound) {
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
    return make_pair(lbound, dist_l);
  }
  if (/*dist_l == DBL_MAX || */ dist_r <= dist_l) {
    rbound++;
    return make_pair(rbound, dist_r);
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
    DBG(cout << ">processing " << obj_idx << " (" << (int)nodeinfo[obj_idx].visited <<", " << nodeinfo[obj_idx].cluster <<")" << endl;)
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
    DBG( cout << "Expanding cluster " << current_cluster << " with " << neighbours.size() << " neighbours:" << endl;)
    for (size_t j = 0; j < neighbours.size(); ++j) {
      size_t nb_obj_idx = reference_distances[neighbours[j]].second;
      DBG(cout << "  processing " << nb_obj_idx << " (" << (int)nodeinfo[nb_obj_idx].visited <<", " << nodeinfo[nb_obj_idx].cluster <<")" << endl;)
      if (nodeinfo[nb_obj_idx].visited == false) {
        nodeinfo[nb_obj_idx].visited = true;
        std::vector<size_t> nneighbours;
        getneighbours(reference_distances, dsm, nneighbours, neighbours[j], eps);
        if (nneighbours.size()+1 >= minpts) {
          neighbours.insert(neighbours.end(), nneighbours.begin(), nneighbours.end());
          DBG(cout << "  Added " << nneighbours.size() << " new neighbours" << endl;)
        }
      }
      if (nodeinfo[nb_obj_idx].cluster == "") {
        nodeinfo[nb_obj_idx].cluster = current_cluster_s;
        current_cluster_size++;
        nodeinfo[nb_obj_idx].noise=false;
      }
    }
    DBG( cout << "  Cluster size: " << current_cluster_size << endl;)
  }
}

/*}}}*/
/* membership tests {{{*/
/**
 * On API design: "in a good API, you can guess the purpose of each parameter just by looking at the
 * type signature, ignoring the variable names"
 *
 * clearly not the case here.
 */
template<class mt>
void test_epsneighbourhood(typename mt::type& data, typename mt::type &test_data,
    bool use_triangle_ineq, size_t reference_point, distvector_t& reference_distances_unsorted,
    featurescale_t &featurescale, const std::vector<node_meta> &nodemeta,
    double eps, bool collect_neighbours, std::vector<node_meta> &nodemetaout) {
  distvector_t result_indizes;
  nodemetaout.resize(test_data.n_rows);

  for (size_t i = 0; i < test_data.n_rows; ++i) {
    progress("testing EPS neighbourhood", i, test_data.n_rows);
    double testpoint_to_reference = 0;
    result_indizes.clear();
    if (use_triangle_ineq) {
       testpoint_to_reference = mt::calculate_distance(data, reference_point, test_data, i, featurescale);
    }

    for (size_t j = 0; j < data.n_rows; ++j) {
      // TODO make this use get_next as well
      if (use_triangle_ineq && (fabs(reference_distances_unsorted[j].first - testpoint_to_reference) > eps)) {
        continue;
      }
      double dist = mt::calculate_distance(test_data, i, data, j, featurescale);
      if (dist <= eps) {
        result_indizes.push_back(make_pair(dist, j));
      }
    }

    map<string, size_t> occurences;
    for (size_t j = 0; j < result_indizes.size(); ++j) {
      string cl = nodemeta[result_indizes[j].second].cluster;
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
    if (collect_neighbours) {
      nodemetaout[i].neighbours = result_indizes;
    }
    nodemetaout[i].cluster = cur_max_lbl;
  }
} 

template<class mt>
void test_knneighbourhood(typename mt::type& data, typename mt::type &test_data, bool use_triangle_ineq, size_t reference_point, 
    distvector_t& reference_distances_sorted, featurescale_t &featurescale,
    const std::vector<node_meta> &nodemeta, size_t k, bool collect_neighbours, std::vector<node_meta> &nodemetaout) {
  /* TODO triangle inequality:
   * for (t test_data) {
   *   this_ref_distance = distance(reference_point, t);
   *   distvector_t result_indizes(k);
   *   for (d = getnext(data)) {
   *     if (fabs(reference_distances(d)-this_ref_distance) > eps)
   *       break;
   *     real_dist = distance(t, d);
   *     if (real_dist < result_indizes[result_indizes.size()-1].first) {
   *       result_indizes[result_indizes.size()-1] = make_pair(real_dist, d)
   *       sort(result_indizes.begin(), result_indizes.end());
   *     }
   *   } // rest as below
   * }
   */
  distvector_t result_indizes;
  nodemetaout.resize(test_data.n_rows);
  double testpoint_to_reference = 0.0;
  
  for (size_t i = 0; i < test_data.n_rows; ++i) {
    progress("testing k-nearest neighbours", i, test_data.n_rows);
    result_indizes.clear();
    result_indizes.reserve(k);
    if (use_triangle_ineq) {
      testpoint_to_reference = mt::calculate_distance(data, reference_point, test_data, i, featurescale);
      size_t l = 0, r = reference_distances_sorted.size();
      // 0 1 2 4 5
      // -> 3
      size_t m = 0;
      while (l != r) {
        m = (l+r)/2;
        if (testpoint_to_reference < reference_distances_sorted[m].first) {
          r = m; 
        } else if (reference_distances_sorted[m].first < testpoint_to_reference) {
          if (l == m)
            break;
          l = m;
        } else {
          // we have found the exact point.
          break;
        }
      }
      if (m == l && r == m+1) {
        if (fabs(testpoint_to_reference-reference_distances_sorted[r].first) < fabs(testpoint_to_reference-reference_distances_sorted[m].first)) {
          m = r;
        }
      }
      // this test always returns the self point as nearest neighbour, if it is in the data set.
      // if the latter is the case, return it:
      if (reference_distances_sorted[m].first == testpoint_to_reference) {
        result_indizes.push_back(make_pair(0.0, reference_distances_sorted[m].second));
      }
      // closest pooint = reference_distances_sorted[m]
      size_t lbound=m, rbound=m;
      double mid_value = reference_distances_sorted[m].first; 
      while (true) {
        pair<unsigned int, double> closest_object = get_next(reference_distances_sorted, mid_value, lbound, rbound);
        if (closest_object.first == UINT_MAX) {
          break;
        }
        if (result_indizes.size() == k && result_indizes[k-1].first < closest_object.second) {
          // we've found or k nearest members, the next closest objects 'minimum distance' is greater than the
          // real distance of the k-th farthest element that we have already calculated.
          break;
        }

        size_t real_index = reference_distances_sorted[closest_object.first].second;
        double dist = mt::calculate_distance(data, i, test_data, real_index, featurescale);
        
        if (result_indizes.size() == k && result_indizes[k-1].first <= dist) {
          continue;
        }

        if (result_indizes.size() < k) {
          result_indizes.push_back(make_pair(dist, real_index));
          if (result_indizes.size() == k) {
            sort(result_indizes.begin(), result_indizes.end());
          }
        } else {
          result_indizes[k-1] = make_pair(dist, real_index);
          sort(result_indizes.begin(), result_indizes.end());
        }
      }

    } else { 
      for (size_t j = 0; j < data.n_rows; ++j) {
        double dist = mt::calculate_distance(test_data, i, data, j, featurescale);
        result_indizes.push_back(make_pair(dist, j));
      }
      sort(result_indizes.begin(), result_indizes.end());
    }

    if (collect_neighbours) {
      nodemetaout[i].neighbours.reserve(result_indizes.size());
    }
    size_t cur_max = 0;
    string cur_max_lbl;

    map<string, size_t> occurences;
    for (size_t j = 0; j < result_indizes.size() && j < k; ++j) {
      string cl = nodemeta[result_indizes[j].second].cluster;
      occurences[cl]++;
      if (collect_neighbours) {
        nodemetaout[i].neighbours.push_back(result_indizes[j]);
      }
      // k+1: if the cluster is ambiguous, collect more neighbours.
      // Check if we have reached k (assuming that we have more than k neighbours)
      if (j < k && j+1 < result_indizes.size()) {
        continue;
      }
      std::vector<ssize_t> hist;
      for (std::map<string, size_t>::iterator it = occurences.begin(); it != occurences.end(); ++it) {
        hist.push_back(it->second);
      }
      if (hist.size() < 2) {
        break;
      }
      sort(hist.begin(), hist.end());
      if (hist[hist.size()-2] != hist[hist.size()-1]) {
        break;
      }
    }

    for (std::map<string, size_t>::iterator it = occurences.begin(); it != occurences.end(); ++it) {
      if (it->second > cur_max) {
        cur_max_lbl = it->first;
        cur_max = it->second;
      }
    }

    if (!occurences.size()) {
      nodemetaout[i].noise = true;
    }
    nodemetaout[i].cluster = cur_max_lbl;
  }
} 
/* }}} */
/* cluster options & command line parsing {{{ */
namespace po = boost::program_options;

class options {
  public:
    bool use_triangle_inequality;
    bool use_feature_scaling ;
    int norm;
    string cluster_method;
    string filetype;
    string infile;
    double eps;
    size_t minpts;
    string test_membership;
    int test_membership_k;
    double test_membership_eps;
    string test_membership_filename;
    string label_file;
    string test_write_neighbours_file;
    bool dump_distancematrix;

    options() {
      use_triangle_inequality = true;
      use_feature_scaling = false;
      dump_distancematrix = false;
    }
    int init(int argc, char ** argv) { 
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
        ("cluster-method,c", po::value<string>()->default_value("none"), "Cluster method. Possible values are\n"
                                                                          "  dbscan - cluster using dbscan\n"
                                                                          "  file   - read labels from file\n"
                                                                          "  none   - don't perform clustering\n")
        ("labels,l", po::value<string>()->default_value("no"), "labels file (required for --cluster-method file)")
        ("test", po::value<string>()->default_value("no"), "test membership method, possible values:\n"
                                                "  no  - don't test membership\n"
                                                "  eps - use eps neighbourhood\n"
                                                "  k   - use k-nearest-neighbours")
        ("test-file", po::value<string>(), "file to read membership test values from. If not set, the main input file will be used.")
        ("test-parameter", po::value<double>(), "membership test parameter (eps or k)")
        ("test-neighbours-out", po::value<string>()->default_value(""), "write neighbours to <file>")
        ("dump-distancematrix", "dump distancematrix to distancematrix.out")
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
      norm = vm["norm"].as<int>();
      if (norm != 1 && norm != 2) {
        cerr << "Only norms 1 or 2 are supported" << endl;
        cerr << desc << endl;
        return 1;
      }

      cluster_method = vm["cluster-method"].as<string>();
    
      if (cluster_method == "file") {
        if (!vm.count("labels")) {
          cerr << "Error: --cluster-method=file requires a --labels file" << endl;
          cerr << desc << endl;
          return 1;
        }
        label_file = vm["labels"].as<string>();
      } else if (cluster_method != "dbscan" && cluster_method != "none") {
        cerr << "Error: invalid value for --cluster-method" << endl;
        cerr << desc << endl;
        return 1;
      }

      filetype = vm["input-type"].as<string>();

      infile = vm["input-file"].as<string>();
      eps = vm["epsilon"].as<double>();
      minpts = vm["minpts"].as<unsigned int>();

      test_membership = vm["test"].as<string>();
      test_membership_k = 0;
      test_membership_eps = 0.0;
      test_membership_filename = "";
      test_write_neighbours_file = vm["test-neighbours-out"].as<string>();

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
          test_membership_filename = infile; 
        } else {
          test_membership_filename = vm["test-file"].as<string>();
        }
      }

      if (vm.count("dump-distancematrix")) {
        dump_distancematrix = true;
      }
      return 0;
    }
};
/*}}}*/
/* main {{{*/

template<class mt>
int run_clusterer(const options& o) {
  trace("load file");
  typename mt::type data;
  featurescale_t featurescale;
  try {
    mt::load_file(o.infile, data, featurescale);
  } catch (const exception& c) {
    cout << "Error: " << c.what() << endl;
    return 1;
  }

  if (!o.use_feature_scaling) {
    featurescale.clear();
    featurescale.insert(featurescale.end(), data.n_cols, 1.0);
  }

  cout << "Input matrix size is: " << data.n_rows << "x" << data.n_cols << endl;

  // basepoints for triangle inequality
  distvector_t reference_distances;
  reference_distances.resize(data.n_rows);

  if (o.use_triangle_inequality) {
    trace("create reference distances");
    for (unsigned int i = 0; i < data.n_rows; ++i) {
      progress("creating reference distances", i, data.n_rows);
      reference_distances[i] = make_pair(mt::calculate_distance(data, 0, data, i, featurescale), i);
      DBG(cout << reference_distances[i].first << " ";)
    }
    DBG(cout << endl;)
  } else {
    for (unsigned int i = 0; i < reference_distances.size(); ++i) {
      reference_distances[i] = make_pair(0.0, i);
    }
  }


  std::vector<node_meta> metadata(data.n_rows);
  if (o.cluster_method == "dbscan") {
    trace("create distancematrix");
    distancematrix_t distancematrix(data.n_rows, data.n_rows);
    create_distance_matrix<mt>(data, reference_distances, o.eps, featurescale, distancematrix);
    distancematrix = symmatu(distancematrix);
    
    trace("sort reference distances");
    distvector_t reference_distances_sorted = reference_distances;
    if (o.use_triangle_inequality) {
      sort(reference_distances_sorted.begin(), reference_distances_sorted.end());
    }

    trace("DBSCAN");
    dbscan(reference_distances_sorted, distancematrix, o.eps, o.minpts, metadata);
    if (o.dump_distancematrix) {
      trace("dumping distance matrix");
      FILE *f = fopen("distmatrix.out", "w");
      bool first;
      for (size_t i = 0; i < distancematrix.n_rows; ++i) {
        first = true;
        for (size_t j = 0; j < distancematrix.n_cols; ++j) {
          if (first) {
            fprintf(f, "%f", distancematrix(i,j));
            first = false;
          } else {
            fprintf(f, " %f", distancematrix(i,j));
          }
        }
        fputs("\n", f);
      }
      fclose(f);
    }

    trace("DBSCAN output");
    {
      FILE *f = fopen("cluster.out", "w");
      for (size_t i = 0; i < metadata.size(); ++i) {
        fprintf(f, "%s%s\n", metadata[i].cluster.c_str(), metadata[i].noise?" (noise)":"");
      }
      fclose(f);
    }
  } else if (o.cluster_method == "file") {
    trace("label-load");
    load_file_clusters(o.label_file, metadata);
  } else if (o.cluster_method != "none"){
    throw runtime_error("unsupported clustering method " + o.cluster_method);
  }

  // Test membership if wanted
  if (o.test_membership != "no") {
    trace("loading test file");
    typename mt::type test_data;
    featurescale_t test_featurescale;
    try {
      if (o.filetype == "cluto") {
        mt::load_file(o.test_membership_filename, test_data, test_featurescale);
      } else {
        mt::load_file(o.test_membership_filename, test_data, test_featurescale);
      }
    } catch (const runtime_error& e) {
      cerr << "Error while loading test member ship data: " << e.what() << endl;
      return 1;
    }
    
    if (o.test_membership == "eps" || o.test_membership == "k") {
      trace("neighbourhood test");
      bool write_neighbours = false;
      if (o.test_write_neighbours_file.size()) {
        write_neighbours = true;
      }
      std::vector<node_meta> nodemetaout;
      if (o.test_membership == "eps") {
        test_epsneighbourhood<mt>(data, test_data, 
              o.use_triangle_inequality, 0u, reference_distances, featurescale, metadata,
              o.test_membership_eps, write_neighbours, nodemetaout); 
      } else {
        distvector_t reference_distances_sorted = reference_distances;
        sort(reference_distances_sorted.begin(), reference_distances_sorted.end());
        test_knneighbourhood<mt>(data, test_data,
            o.use_triangle_inequality, 0u, reference_distances_sorted, featurescale, metadata,
            o.test_membership_k, write_neighbours, nodemetaout); 
      }

      trace("writing membership.out");
      {
        FILE *f = fopen("membership.out", "w");
        for (size_t i = 0; i < nodemetaout.size(); ++i) {
          fprintf(f, "%s%s\n", nodemetaout[i].cluster.c_str(), nodemetaout[i].noise?" (noise)":"");
        }
        fclose(f);
      }

      if (write_neighbours) {
        trace("writing neighbour information");
        FILE *f = fopen(o.test_write_neighbours_file.c_str(), "w");
        for (size_t i = 0; i < nodemetaout.size(); ++i) {
          bool first = true;
          for (distvector_t::iterator it = nodemetaout[i].neighbours.begin();
                it != nodemetaout[i].neighbours.end(); ++it)  { 
            if (first) {
              first = false;
              fprintf(f, "%lu[%f]", it->second, it->first);
            } else {
              fprintf(f, " %lu[%f]", it->second, it->first);
            }
          }
          fprintf(f, "\n");
        }
        fclose(f);
      }
    }
  }

  progress("", 0, 0);
  dump_trace();

  return 0;
}


int main(int argc, char ** argv) {
  options o;
  try {
    int ret = o.init(argc, argv);
    if (ret) 
      return ret;
  } catch (const std::runtime_error& e) {
    cerr << e.what() << endl;
    return 1;
  }

  cout << "reading from " << o.infile << ", eps=" << o.eps <<", minpts=" << o.minpts << ", "
    << (o.use_triangle_inequality?"":"not") << " using triangle inequality, "
    << (o.use_feature_scaling?"":"not") << " using feature scaling." << endl;

  try {
    int ret;
    if (o.filetype == "cluto") {
      if (o.norm == 1) {
        ret = run_clusterer<mt_sparse_simple<1> >(o);
      } else {
        ret = run_clusterer<mt_sparse_simple<2> >(o);
      }
    } else {
      if (o.norm == 1) {
        ret = run_clusterer<mt_dense<1> >(o);
      } else {
        ret = run_clusterer<mt_dense<2> >(o);
      }
    }
    if (ret) {
      return 1;
    }
  } catch (const runtime_error& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
}
/*}}}*/
/* vim: set fdm=marker: */
