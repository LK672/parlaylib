#include <atomic>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <parlay/delayed.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "helper/ligra_light.h"
#include "helper/union_find.h"

/**
 * source:
 *   Gary L. Miller, Richard Peng, Adrian Vladu, and Shen Chen Xu. 2015. 
 *   Improved Parallel Algorithms for Spanners and Hopsets. In Proceedings 
 *   of the 27th ACM Symposium on Parallelism in Algorithms and Architectures 
 *   (Portland, Oregon, USA) (SPAA ’15).
 */

// **************************************************************
// Exponential Start Time Clustering (ESTC)
// Returns a spanning forest and a set of inter-cluster edges.
// Each vertex v is assigned to cluster u = arg min { dist(u, v) - delta_u }
// where delta_u is drawn from an exponential distribution.
// **************************************************************
template <typename vertex>
auto ESTC_with_edges(float beta,
                     const parlay::sequence<parlay::sequence<vertex>>& G,
                     const parlay::sequence<parlay::sequence<vertex>>& GT) {
  long n = G.size();
  parlay::random_generator g(0);
  std::exponential_distribution<float> exp(beta);

  // Generate exponential shifts and bucket them
  auto exps = parlay::tabulate(n, [&](long i) {
    auto r = g[i];
    return (int)std::floor(exp(r));
  });
  int max_e = parlay::reduce(exps, parlay::maximum<int>());
  auto buckets = parlay::group_by_index(
      parlay::delayed::tabulate(n, [&](vertex i) {
        return std::pair(max_e - exps[i], i);
      }),
      max_e + 1);

  // labels[v] stores the center of the cluster containing v
  auto labels = parlay::tabulate<std::atomic<vertex>>(n, [](long i) { return -1; });
  // parents[v] stores the parent of v in the spanning forest
  auto parents = parlay::tabulate<std::atomic<vertex>>(n, [](long i) { return -1; });

  auto edge_f = [&](vertex u, vertex v) -> bool {
    vertex expected = -1;
    if (labels[v].compare_exchange_strong(expected, labels[u])) {
      parents[v] = u;
      return true;
    }
    return false;
  };
  auto cond_f = [&](vertex v) { return labels[v] == -1; };
  auto frontier_map = ligra::edge_map(G, GT, edge_f, cond_f);

  ligra::vertex_subset<vertex> frontier;
  for (int i = 0; i <= max_e; i++) {
    frontier.add_vertices(parlay::filter(buckets[i], [&](vertex v) {
      if (labels[v] != -1) return false;
      labels[v] = v;
      return true;
    }));
    frontier = frontier_map(frontier);
  }

  auto labels_final = parlay::map(labels, [](auto& l) { return l.load(); });

  // Extract forest edges
  auto forest_edges = parlay::filter(
      parlay::delayed_tabulate(n, [&](vertex v) {
        return std::pair(parents[v].load(), v);
      }),
      [](auto p) { return p.first != -1; });

  // Extract inter-cluster edges: one edge from each boundary vertex to each adjacent cluster
  auto inter_cluster_edges = parlay::flatten(parlay::tabulate(n, [&](vertex v) {
    auto neighbors = G[v];
    auto labeled_neighbors = parlay::filter(
        parlay::map(neighbors, [&](vertex u) {
          return std::pair(labels_final[u], u);
        }),
        [&](auto p) { return p.first != labels_final[v]; });
    
    if (labeled_neighbors.empty()) return parlay::sequence<std::pair<vertex, vertex>>();
    
    auto sorted_ngh = parlay::sort(labeled_neighbors);
    auto unique_indices = parlay::pack_index<long>(parlay::delayed_tabulate(sorted_ngh.size(), [&](long i) {
      return (i == 0 || sorted_ngh[i].first != sorted_ngh[i-1].first);
    }));
    
    return parlay::map(unique_indices, [&](long i) {
      return std::pair(v, sorted_ngh[i].second);
    });
  }));

  return std::make_pair(std::move(forest_edges), std::move(inter_cluster_edges));
}

// **************************************************************
// Weighted Spanner
// Implements Algorithm 3 (WellSeparatedSpanner) from Miller et al.
// Returns a sequence of edges (as triples: u, v, weight) that form the spanner.
// **************************************************************
template <typename vertex, typename weight>
auto weighted_spanner(const parlay::sequence<parlay::sequence<std::pair<vertex, weight>>>& G, int k) {
  long n = G.size();
  float beta = std::log((double)n) / (2.0 * k);

  // Use a pair of pairs instead of tuple to leverage parlay::hash<pair>
  using edge_t = std::pair<std::pair<vertex, vertex>, weight>;

  // Collect all edges and normalize them to u < v
  auto all_edges = parlay::flatten(parlay::tabulate(n, [&](vertex u) {
    return parlay::map(G[u], [=](auto p) {
      vertex v = p.first;
      if (u < v) return edge_t(std::pair(u, v), p.second);
      else return edge_t(std::pair(v, u), p.second);
    });
  }));
  
  // Deduplicate input edges
  auto unique_edges = parlay::remove_duplicates(all_edges);
  
  if (unique_edges.empty()) return unique_edges;

  auto min_w = parlay::reduce(parlay::delayed_map(unique_edges, [](auto e) { return e.second; }), parlay::minimum<weight>());
  auto max_w = parlay::reduce(parlay::delayed_map(unique_edges, [](auto e) { return e.second; }), parlay::maximum<weight>());
  
  // 1. compute the edge weight buckets in order
  int num_buckets = (min_w == max_w) ? 1 : (int)std::floor(std::log2((double)max_w / min_w)) + 1;
  auto buckets = parlay::group_by_index(parlay::delayed_map(unique_edges, [=](auto e) {
    weight w = e.second;
    int b = (w == 0 || min_w == max_w) ? 0 : (int)std::floor(std::log2((double)w / min_w));
    return std::pair(std::min(b, num_buckets - 1), e);
  }), num_buckets);

  // 2. initialize H and S
  parlay::sequence<edge_t> S;
  union_find<vertex> UF(n);
  
  // 3. loop through all the buckets
  for (auto& bucket_edges : buckets) {
    if (bucket_edges.empty()) continue;
    
    // map to active components of H
    auto comp_id = parlay::tabulate(n, [&](vertex i) { return UF.find(i); });
    auto component_map = parlay::remove_duplicates(comp_id);
    auto rev_comp_map = parlay::sequence<vertex>(n, -1);
    parlay::parallel_for(0, component_map.size(), [&](long i) {
      rev_comp_map[component_map[i]] = i;
    });
    
    // 4. Compute Gamma_i (take only bucket_edges that are not part of the forest edges)
    long num_comps = component_map.size();
    if (num_comps == 0) continue;
    auto gamma_edges = parlay::filter(parlay::map(bucket_edges, [&](auto e) {
      return std::pair(rev_comp_map[comp_id[e.first.first]], rev_comp_map[comp_id[e.first.second]]);
    }), [](auto p) { return p.first != p.second; });
    
    if (gamma_edges.empty()) continue;
    
    auto Gamma = parlay::group_by_index(gamma_edges, num_comps);

    // 5. Call ESTC on Gamma_i
    // Note: since the edge weights are uniform when we do ESTC, we don't
    // have to consider the edge weights, an approximate dist(u, v) as
    // uniform across all pairs of vertices used
    auto [forest, inter] = ESTC_with_edges(beta, Gamma, Gamma);
    
    // need to get back original edges from result of ESTC
    auto edge_to_original = parlay::sort(parlay::map(bucket_edges, [&](auto e) {
      vertex u = rev_comp_map[comp_id[e.first.first]];
      vertex v = rev_comp_map[comp_id[e.first.second]];
      if (u > v) std::swap(u, v);
      return std::make_pair(std::pair(u, v), e);
    }));
    
    auto find_original = [&](vertex u, vertex v) {
      if (u > v) std::swap(u, v);
      auto pair = std::pair(u, v);
      auto it = std::lower_bound(edge_to_original.begin(), edge_to_original.end(), 
                                 std::make_pair(pair, edge_to_original[0].second),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
      return it->second;
    };
    
    auto s_forest = parlay::map(forest, [&](auto p) {
      auto e = find_original(p.first, p.second);
      // 7. Add forest F to H 
      UF.link(e.first.first, e.first.second);
      return e;
    });
    
    auto s_inter = parlay::map(inter, [&](auto p) {
      return find_original(p.first, p.second);
    });
    
    // 7. Add forst F to S
    S = parlay::append(S, s_forest);

    // 8. Add edges from boundary vertices to adjacent clusters to S
    S = parlay::append(S, s_inter);
  }
  
  // 9. return result
  return S;
}
