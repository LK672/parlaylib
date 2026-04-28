#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include <parlay/delayed.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/random.h>

#include "weighted_spanner.h"

/**
 * source:
 *   Ioannis Koutis. 2014. Simple parallel and distributed algorithms for 
 *   spectral graph sparsification 
 */

// **************************************************************
// t-Bundle Spanner Construction
// Iteratively computes a spanner and removes its edges from G.
// **************************************************************
template <typename vertex, typename weight>
auto t_bundle_spanner(parlay::sequence<parlay::sequence<std::pair<vertex, weight>>> G, int t, int k) {
  long n = G.size();
  parlay::sequence<std::pair<std::pair<vertex, vertex>, weight>> bundle_edges;

  for (int i = 0; i < t; i++) {
    long current_m = parlay::reduce(parlay::map(G, parlay::size_of())) / 2;
    if (current_m == 0) break;

    // Compute a single spanner on the current graph
    auto H_i = weighted_spanner(G, k);
    bundle_edges = parlay::append(bundle_edges, H_i);

    auto normalize = [](auto e) {
        auto [u, v] = e.first;
        if (u > v) std::swap(u, v);
        return std::pair(u, v);
    };

    // Sort the current spanner edges to allow for fast binary-search filtering
    auto sorted_H = parlay::sort(parlay::map(H_i, normalize));

    // Remove H_i from G using a parallel filter
    G = parlay::tabulate(n, [&](long u) {
      // do sequentially if size is small enough, otherwise in parallel
      if (G[u].size() < 100) {
        std::vector<std::pair<vertex, weight>> kept;
        kept.reserve(G[u].size());
        for (const auto& p : G[u]) {
          vertex v = p.first;
          auto [a, b] = std::minmax((vertex)u, v);
          auto target = std::pair(a, b);
          
          auto it = std::lower_bound(sorted_H.begin(), sorted_H.end(), target);
          if (!(it != sorted_H.end() && *it == target)) {
            kept.push_back(p);
          }
        }
        return parlay::to_sequence(kept);
      } else {
        return parlay::filter(G[u], [&](auto p) {
          vertex v = p.first;
          auto [a, b] = std::minmax((vertex)u, v);
          auto target = std::pair(a, b);
          
          auto it = std::lower_bound(sorted_H.begin(), sorted_H.end(), target);
          return !(it != sorted_H.end() && *it == target); 
        });
      }
    });
  }
  return bundle_edges;
}

// **************************************************************
// Algorithm 1: HALFSPARSIFY
// Reduces the graph size by half while preserving spectral properties.
// **************************************************************
template <typename vertex, typename weight>
auto half_sparsify(const parlay::sequence<parlay::sequence<std::pair<vertex, weight>>>& G, float epsilon, int k = 1) {
  long n = G.size();

  if (n == 0) return G;
  
  // Calculate t
  float log_n = std::log2(n);
  int t = std::ceil((48.0 * log_n * log_n) / (epsilon * epsilon));
  // Only for testing purposes so that running sparsify doesn't take too long:
  // t = 4;

  // Compute the t-bundle spanner
  auto bundle_edges = t_bundle_spanner(G, t, k);

  // Create a sorted lookup sequence of the bundle edges
  auto sorted_bundle = parlay::sort(parlay::map(bundle_edges, [](auto e) {
    vertex u = e.first.first;
    vertex v = e.first.second;
    if (u > v) std::swap(u, v);
    return std::pair(u, v);
  }));

  // Extract all unique undirected edges from the original graph G
  return parlay::tabulate(n, [&](long u) {
    // do sequentially if size is small enough, otherwise in parallel
    if (G[u].size() < 100) {
      std::vector<std::pair<vertex, weight>> kept;
      for (const auto& p : G[u]) {
        vertex v = p.first;
        auto [a, b] = std::minmax((vertex)u, v);
        auto target = std::pair(a, b);

        auto it = std::lower_bound(sorted_bundle.begin(), sorted_bundle.end(), target);
        bool in_H = (it != sorted_bundle.end() && *it == target);

        if (in_H) {
          kept.push_back(p);
        } else {
          parlay::random_generator gen((size_t)a * n + b);
          auto r = gen[0];
          std::uniform_real_distribution<float> dist(0.0, 1.0);
          if (dist(r) <= 0.25f) {
            kept.push_back({v, p.second * 4});
          }
        }
      }
      return parlay::to_sequence(kept);
    } else {
      auto sampled_row = parlay::filter(parlay::delayed_map(G[u], [&](auto p) -> std::optional<std::pair<vertex, weight>> {
        vertex v = p.first;
        auto [a, b] = std::minmax((vertex)u, v);
        auto target = std::pair(a, b);

        auto it = std::lower_bound(sorted_bundle.begin(), sorted_bundle.end(), target);
        bool in_H = (it != sorted_bundle.end() && *it == target);

        if (in_H) {
          return std::optional<std::pair<vertex, weight>>(p);
        } else {
          parlay::random_generator gen((size_t)a * n + b); 
          auto r = gen[0];
          std::uniform_real_distribution<float> dist(0.0, 1.0);
          if (dist(r) <= 0.25f) {
            return std::optional<std::pair<vertex, weight>>({v, p.second * 4});
          } else {
            return std::nullopt;
          }
        }
      }), [](const auto& opt) { return opt.has_value(); });

      return parlay::map(sampled_row, [](const auto& opt) { return *opt; });
    }
  });
}

// **************************************************************
// Algorithm 3: SPARSIFY
// Iteratively halves the graph to reach a target sparsification factor rho.
// **************************************************************
template <typename vertex, typename weight>
auto sparsify(parlay::sequence<parlay::sequence<std::pair<vertex, weight>>> G, float epsilon, float rho, int k = 1) {
  long n = G.size();
  long m = parlay::reduce(parlay::map(G, parlay::size_of())) / 2;

  if (n == 0 || m == 0) return G;

  // Calculate threshold
  float log_n = std::log2(n);
  long threshold = std::ceil((48.0 * n * log_n * log_n) / (epsilon * epsilon));

  // If graph already sparse, don't sparsify
  // If testing on small t values, then you should comment the threshold checks
  if (m <= threshold) return G; 

  int iterations = (rho > 0) ? std::ceil(std::log2(rho)) : 0;
  if (iterations <= 0) return G;
  float local_epsilon = epsilon / iterations;

  for (int i = 0; i < iterations; i++) {
    long current_m = parlay::reduce(parlay::map(G, parlay::size_of())) / 2;
    if (current_m <= threshold) break;

    G = half_sparsify(G, local_epsilon, k);
  }
  
  return G;
}
