#include <iostream>
#include <string>
#include <cmath>

#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/internal/get_time.h>

#include "sdd_solver.h"
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
// Driver for Parallel SDD Solver
// **************************************************************
int main(int argc, char* argv[]) {
  using vertex = int;
  using weight = double;
  using utils = graph_utils<vertex>;

  auto usage = "Usage: sdd_solver <n> [depth] [epsilon] [rho]";
  if (argc < 2) {
    std::cout << usage << std::endl;
    return 1;
  }

  long n = 0;
  int d = 3;            // Default depth of the approximate inverse chain (log kappa)
  float epsilon = 0.5f; // Default sparsifier tolerance
  float rho = 4.0f;     // Default sparsification target factor
  int num_runs = 5;

  try { n = std::stol(argv[1]); }
  catch (...) {
    std::cout << "Invalid n" << std::endl;
    return 1;
  }
  if (argc > 2) {
    try { d = std::stoi(argv[2]); }
    catch (...) {}
  }
  if (argc > 3) {
    try { epsilon = std::stof(argv[3]); }
    catch (...) {}
  }
  if (argc > 4) {
    try { rho = std::stof(argv[4]); }
    catch (...) {}
  }

  // -------------------------------------------------------------
  // SETUP: Generate the SDD Matrix M = D - A
  // -------------------------------------------------------------
  // Generate a random symmetric graph with roughly 10 edges per vertex
  long target_m = 10 * n; 
  auto G_unweighted = utils::rmat_symmetric_graph(n, target_m);
  auto A = utils::add_weights<weight>(G_unweighted, 1.0, 10.0);

  n = A.size();

  // Build the Diagonal matrix D
  parlay::sequence<weight> D = parlay::tabulate(n, [&](long i) {
    weight degree_sum = 0;
    for (const auto& edge : A[i]) {
      degree_sum += edge.second;
    }
    // ADD 0.1: This makes the matrix *strictly* diagonally dominant. 
    // It removes the Laplacian null-space, ensuring the matrix is perfectly invertible.
    return degree_sum + 0.1; 
  });

  SDD_Matrix<vertex, weight> M_0 = {A, D};

  // Create an arbitrary target vector 'b' to solve for (M * x = b)
  parlay::sequence<weight> b_target = parlay::tabulate(n, [](long i) { 
    return (double)((i % 100) + 1) / 10.0; 
  });

  std::cout << "--- SDD Matrix System ---" << std::endl;
  std::cout << "Variables (n) : " << n << std::endl;
  std::cout << "Chain Depth   : " << d << std::endl;
  std::cout << "Sparsifier    : epsilon = " << epsilon << ", rho = " << rho << std::endl;
  std::cout << "-------------------------" << std::endl;

  parlay::internal::timer t("Time");

  // -------------------------------------------------------------
  // PHASE 1: PRECOMPUTATION (Chain Building)
  // -------------------------------------------------------------
  std::cout << "\n[Phase 1] Building Approximate Inverse Chain..." << std::endl;
  
  parlay::internal::timer t_build("Time", false);
  t_build.start();
  int k = std::max(1, (int)parlay::log2_up(n));
  auto cached_chain = build_inverse_chain(M_0, d, epsilon, rho, k);
  double build_time = t_build.stop();
  
  std::cout << "Chain build time: " << build_time << " seconds" << std::endl;

  // -------------------------------------------------------------
  // PHASE 2: SOLVE (Preconditioned Richardson Iteration)
  // -------------------------------------------------------------
  std::cout << "\n[Phase 2] Executing Solver..." << std::endl;
  
  weight tolerance = 1e-6; // Target L2 residual error
  int max_iters = 100;     // Fallback limit

  std::vector<double> solve_times;
  parlay::sequence<weight> x_solution;
  parlay::internal::timer t_solve("Time", false);
 
  for (int run = 0; run < num_runs; run++) {
    t_solve.start();
    x_solution = sdd_solve(M_0, b_target, cached_chain, tolerance, max_iters);
    double elapsed = t_solve.stop();
    solve_times.push_back(elapsed);
    std::cout << "Run " << (run + 1) << ": " << elapsed << " seconds" << std::endl;
  }
 
  // Compute statistics
  auto [mean, stddev] = compute_stats(solve_times);
  double std_err = stddev / std::sqrt(num_runs);
  double confidence_95 = 1.96 * std_err;

  // -------------------------------------------------------------
  // VERIFICATION: Check the actual residual (b - M * x)
  // -------------------------------------------------------------
  auto D_x = parlay::tabulate(n, [&](long j) { return M_0.D[j] * x_solution[j]; });
  auto A_x = multiply_A_x(M_0.A, x_solution);
  auto M_x = vector_sub(D_x, A_x);
  auto residual = vector_sub(b_target, M_x);
  
  weight final_r_norm = std::sqrt(parlay::reduce(parlay::map(residual, [](weight val) { return val * val; })));
  weight initial_b_norm = std::sqrt(parlay::reduce(parlay::map(b_target, [](weight val) { return val * val; })));
 
  std::cout << "\n--- Timing Statistics (Solve Phase) ---" << std::endl;
  std::cout << "Mean time       : " << mean << " seconds" << std::endl;
  std::cout << "Std deviation   : " << stddev << " seconds" << std::endl;
  std::cout << "Std error       : " << std_err << " seconds" << std::endl;
  std::cout << "95% CI          : ± " << confidence_95 << " seconds" << std::endl;
  std::cout << "Range           : [" << (mean - confidence_95) << ", " 
            << (mean + confidence_95) << "]" << std::endl;
 
  std::cout << "\n--- Solver Results ---" << std::endl;
  std::cout << "Initial ||b|| norm : " << initial_b_norm << std::endl;
  std::cout << "Final residual norm: " << final_r_norm << std::endl;
  
  if (final_r_norm <= tolerance) {
    std::cout << "Status             : CONVERGED successfully!" << std::endl;
  } else {
    std::cout << "Status             : FAILED to reach tolerance within max_iters." << std::endl;
  }
 
  std::cout << "\n--- Total Time ---" << std::endl;
  std::cout << "Chain build        : " << build_time << " seconds" << std::endl;
  std::cout << "Avg solve time     : " << mean << " seconds" << std::endl;
  std::cout << "Total (build + avg): " << (build_time + mean) << " seconds" << std::endl;
 
  return 0;
}