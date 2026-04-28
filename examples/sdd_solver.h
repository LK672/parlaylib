#include <cmath>
#include <vector>
#include <utility>

#include <parlay/delayed.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "spectral_sparsifier.h"

/**
 * sources:
 *   Ioannis Koutis. 2014. Simple parallel and distributed algorithms for 
 *   spectral graph sparsification.
 * 
 *   Richard Peng and Daniel A. Spielman. 2014. An efficient parallel solver
 *   for SDD linear systems. In Proceedings of the Forty-Sixth Annual ACM 
 *   Symposium on Theory of Computing (New York, New York) (STOC ’14).
 */

// **************************************************************
// Helper: Parallel Vector Math
// **************************************************************
template <typename weight>
auto vector_add(const parlay::sequence<weight>& v1, const parlay::sequence<weight>& v2) {
  return parlay::tabulate(v1.size(), [&](long i) { return v1[i] + v2[i]; });
}

template <typename weight>
auto vector_sub(const parlay::sequence<weight>& v1, const parlay::sequence<weight>& v2) {
  return parlay::tabulate(v1.size(), [&](long i) { return v1[i] - v2[i]; });
}

template <typename vertex, typename weight>
auto multiply_A_x(const parlay::sequence<parlay::sequence<std::pair<vertex, weight>>>& A,
                  const parlay::sequence<weight>& x) {
  return parlay::tabulate(A.size(), [&](long i) {
    if (A[i].size() < 100) {
      weight sum = 0;
      for (const auto& edge : A[i]) {
        sum += edge.second * x[edge.first];
      }
      return sum;
    } 
    
    return parlay::reduce(parlay::delayed_map(A[i], [&](const auto& edge) {
      return edge.second * x[edge.first];
    }));
  }, 100); 
}

// **************************************************************
// Data Structure for the Approximate Inverse Chain
// **************************************************************
template <typename vertex, typename weight>
struct SDD_Matrix {
  parlay::sequence<parlay::sequence<std::pair<vertex, weight>>> A;
  parlay::sequence<weight> D;
};

// **************************************************************
// Subroutine: Matrix Squaring (A * D^-1 * A)
// Computes the 2-step random walk transitions in parallel.
// **************************************************************
template <typename vertex, typename weight>
auto square_matrix(const SDD_Matrix<vertex, weight>& M) {
  long n = M.A.size();
  parlay::sequence<weight> new_D(n, 0.0);
  auto new_A = parlay::tabulate(n, [&](long u) {
    
    parlay::sequence<std::pair<vertex, weight>> unmerged_edges;

    if (M.A[u].size() < 100) {
      std::vector<std::pair<vertex, weight>> local_edges;
      for (const auto& p1 : M.A[u]) {
        vertex w = p1.first;
        weight A_uw = p1.second;
        for (const auto& p2 : M.A[w]) {
          local_edges.push_back({p2.first, (A_uw * p2.second) / M.D[w]});
        }
      }
      unmerged_edges = parlay::to_sequence(local_edges);
    } 
    else {
      auto delayed_paths = parlay::delayed_map(M.A[u], [&](const auto& p1) {
        vertex w = p1.first;
        weight A_uw = p1.second;
        return parlay::delayed_map(M.A[w], [w, A_uw, &M](const auto& p2) {
          return std::pair<vertex, weight>(p2.first, (A_uw * p2.second) / M.D[w]);
        });
      });
      unmerged_edges = parlay::flatten(delayed_paths);
    }

    // If the unmerged row is huge, sort it in parallel. Otherwise, std::sort.
    if (unmerged_edges.size() > 1000) {
      unmerged_edges = parlay::sort(unmerged_edges, [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
      std::sort(unmerged_edges.begin(), unmerged_edges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Merge duplicate edges
    std::vector<std::pair<vertex, weight>> merged;
    weight self_loop_sum = 0;

    for (const auto& edge : unmerged_edges) {
      if (edge.first == u) {
        self_loop_sum += edge.second; // Accumulate the diagonal
      } else if (!merged.empty() && merged.back().first == edge.first) {
        merged.back().second += edge.second; // Combine weights for duplicate edges
      } else {
        merged.push_back(edge); // New distinct edge
      }
    }
    
    new_D[u] = M.D[u] - self_loop_sum;
    
    return parlay::to_sequence(merged);
  }, 100);

  return SDD_Matrix<vertex, weight>{new_A, new_D};
}

// **************************************************************
// Build the Approximate Inverse Chain
// Recursively squares and sparsifies the matrix d times.
// **************************************************************
template <typename vertex, typename weight>
auto build_inverse_chain(SDD_Matrix<vertex, weight> M_0, int d, float epsilon, float rho, int k) {
  std::vector<SDD_Matrix<vertex, weight>> chain;
  chain.push_back(M_0);

  for (int i = 0; i < d; i++) {
    // Square the matrix algebraically: M_dense = D - A*D^-1*A
    auto M_dense = square_matrix(chain.back());

    // Graph Sparsification: Compress the cliques back down
    auto A_sparse = sparsify(M_dense.A, epsilon, rho, k);

    chain.push_back({A_sparse, M_dense.D});
  }

  return chain;
}

// **************************************************************
// The V-Cycle
// Pushes the vector down the chain, solves the base case, and pulls it up.
// **************************************************************
template <typename vertex, typename weight>
auto v_cycle(const std::vector<SDD_Matrix<vertex, weight>>& chain, const parlay::sequence<weight>& b_0) {
  int d = chain.size() - 1;
  long n = b_0.size();
  
  std::vector<parlay::sequence<weight>> b_vecs;
  b_vecs.push_back(b_0);

  // The Down-Pass: b_i = (I + A_i * D_i^-1) * b_{i-1}
  for (int i = 0; i < d; i++) {
    const auto& M_i = chain[i];
    auto D_inv_b = parlay::tabulate(n, [&](long j) { return b_vecs.back()[j] / M_i.D[j]; });
    auto A_D_inv_b = multiply_A_x(M_i.A, D_inv_b);
    b_vecs.push_back(vector_add(b_vecs.back(), A_D_inv_b));
  }

  // The Base Case: x_d = D_d^-1 * b_d
  parlay::sequence<weight> x_curr = parlay::tabulate(n, [&](long j) {
    return b_vecs.back()[j] / chain.back().D[j];
  });

  // The Up-Pass: x_i = 0.5 * (D_i^-1 * b_i + x_{i+1} + D_i^-1 * A_i * x_{i+1})
  for (int i = d - 1; i >= 0; i--) {
    const auto& M_i = chain[i];
    
    auto D_inv_b = parlay::tabulate(n, [&](long j) { return b_vecs[i][j] / M_i.D[j]; });
    auto A_x = multiply_A_x(M_i.A, x_curr);
    auto D_inv_A_x = parlay::tabulate(n, [&](long j) { return A_x[j] / M_i.D[j]; });
    
    x_curr = parlay::tabulate(n, [&](long j) {
      return 0.5 * (D_inv_b[j] + x_curr[j] + D_inv_A_x[j]);
    });
  }

  return x_curr;
}

// **************************************************************
// Preconditioned Richardson Iteration to drive the residual error to the target tolerance.
// **************************************************************
template <typename vertex, typename weight>
auto sdd_solve(const SDD_Matrix<vertex, weight>& M_0, 
               const parlay::sequence<weight>& b, 
               const std::vector<SDD_Matrix<vertex, weight>>& chain,
               weight tolerance, int max_iters = 100) {
  
  long n = b.size();
  parlay::sequence<weight> x(n, 0.0);

  for (int iter = 0; iter < max_iters; iter++) {
    // Calculate the current prediction: M * x = D * x - A * x
    auto D_x = parlay::tabulate(n, [&](long j) { return M_0.D[j] * x[j]; });
    auto A_x = multiply_A_x(M_0.A, x);
    auto M_x = vector_sub(D_x, A_x);

    // Calculate the residual error: r = b - M * x
    auto r = vector_sub(b, M_x);
    
    // Check for convergence (L2 Norm)
    weight r_norm_sq = parlay::reduce(parlay::map(r, [](weight val) { return val * val; }));
    if (std::sqrt(r_norm_sq) < tolerance) {
      break;
    }

    // Feed the residual error into the V-cycle preconditioner
    auto z = v_cycle(chain, r);

    // Update the guess: x = x + z
    x = vector_add(x, z);
  }

  return x;
}
