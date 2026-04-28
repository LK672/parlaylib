#include <iostream>
#include <string>
#include <tuple>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/get_time.h>

#include "spectral_sparsifier.h"
#include "helper/graph_utils.h"

// Helper to compute mean and standard deviation
std::pair<double, double> compute_stats(const std::vector<double>& times) {
  double sum = 0.0;
  for (double t : times) sum += t;
  double mean = sum / times.size();
  
  double variance = 0.0;
  for (double t : times) {
    double diff = t - mean;
    variance += diff * diff;
  }
  variance /= times.size();
  double stddev = std::sqrt(variance);
  
  return {mean, stddev};
}

// **************************************************************
// Driver for Spectral Sparsifier
// **************************************************************
int main(int argc, char* argv[]) {
  using vertex = int;
  using weight = double;
  using utils = graph_utils<vertex>;
  using weighted_graph = utils::weighted_graph<weight>;

  auto usage = "Usage: spectral_sparsifier <n> [epsilon] [rho]";
  if (argc < 2) {
    std::cout << usage << std::endl;
    return 1;
  }

  long n = 0;
  float epsilon = 0.5f; // Default error tolerance
  float rho = 4.0f;     // Default sparsification target factor

  try { n = std::stol(argv[1]); }
  catch (...) {
    std::cout << "Invalid n" << std::endl;
    return 1;
  }
  if (argc > 2) {
    try { epsilon = std::stof(argv[2]); }
    catch (...) {}
  }
  if (argc > 3) {
    try { rho = std::stof(argv[3]); }
    catch (...) {}
  }

  // Generate a random symmetric graph and add weights
  // Using 20*n edges to ensure it is dense enough to demonstrate sparsification 
  // without blowing up memory on O(N^2) generation for large N.
  long target_m = 1000 * n; 
  auto G_unweighted = utils::rmat_symmetric_graph(n, target_m);
  auto G = utils::add_weights<weight>(G_unweighted, 1.0, 100.0);

  // Calculate actual initial undirected edges
  long initial_directed_edges = parlay::reduce(parlay::map(G, parlay::size_of()));
  long initial_edges = initial_directed_edges / 2;

  std::cout << "--- Initial Graph ---" << std::endl;
  std::cout << "Vertices (n): " << n << std::endl;
  std::cout << "Edges (m)   : " << initial_edges << std::endl;
  std::cout << "Parameters  : epsilon = " << epsilon << ", rho = " << rho << std::endl;
  std::cout << "---------------------" << std::endl;

  int num_runs = 5;
  std::vector<double> times;
  weighted_graph result;
  parlay::internal::timer t("Time");

  // The Koutis-Xu algorithm explicitly requires a log(n) spanner
  int k = std::max(1, (int)std::ceil(std::log2(n)));
  
  // Run spectral sparsifier
  for (int i = 0; i < num_runs; i++) {
    t.start();
    result = sparsify<vertex, weight>(G, epsilon, rho, k);
    double elapsed = t.stop();
    times.push_back(elapsed);
    std::cout << "Run " << (i+1) << ": " << elapsed << " seconds" << std::endl;
  }

  auto [mean, stddev] = compute_stats(times);
  double std_err = stddev / std::sqrt(num_runs);
  double confidence_95 = 1.96 * std_err;  // 95% confidence interval
 
  long final_directed_edges = parlay::reduce(parlay::map(result, parlay::size_of()));
  long final_edges = final_directed_edges / 2;
 
  std::cout << "\n--- Timing Statistics ---" << std::endl;
  std::cout << "Mean time       : " << mean << " seconds" << std::endl;
  std::cout << "Std deviation   : " << stddev << " seconds" << std::endl;
  std::cout << "Std error       : " << std_err << " seconds" << std::endl;
  std::cout << "95% CI          : ± " << confidence_95 << " seconds" << std::endl;
  std::cout << "Range           : [" << (mean - confidence_95) << ", " 
            << (mean + confidence_95) << "]" << std::endl;
 
  std::cout << "\n--- Sparsifier Results ---" << std::endl;
  std::cout << "Final edges      : " << final_edges << " edges" << std::endl;
  std::cout << "Target reduction : " << rho << "x" << std::endl;
  
  if (final_edges > 0) {
    std::cout << "Actual reduction : " << (double)initial_edges / final_edges << "x" << std::endl;
  }
 
  return 0;
}