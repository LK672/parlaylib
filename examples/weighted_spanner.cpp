#include <iostream>
#include <string>
#include <tuple>

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "weighted_spanner.h"
#include "helper/graph_utils.h"

// **************************************************************
// Driver
// **************************************************************
int main(int argc, char* argv[]) {
  using vertex = int;
  using weight = double;
  using utils = graph_utils<vertex>;
  using weighted_graph = utils::weighted_graph<weight>;

  auto usage = "Usage: weighted_spanner <n> [k]";
  if (argc < 2) {
    std::cout << usage << std::endl;
    return 1;
  }

  long n = 0;
  int k = 2; // Default stretch factor parameter
  try { n = std::stol(argv[1]); }
  catch (...) {
    std::cout << "Invalid n" << std::endl;
    return 1;
  }
  if (argc > 2) {
    try { k = std::stoi(argv[2]); }
    catch (...) {}
  }

  // Generate a random symmetric graph and add weights
  auto G_unweighted = utils::rmat_symmetric_graph(n, (n * n) / 4);
  auto G = utils::add_weights<weight>(G_unweighted, 1.0, 100.0);

  utils::print_graph_stats(G_unweighted);
  std::cout << "Stretch parameter k = " << k << std::endl;

  parlay::sequence<std::pair<std::pair<vertex, vertex>, weight>> result;
  parlay::internal::timer t("Time");
  
  // Run weighted spanner
  for (int i = 0; i < 3; i++) {
    result = weighted_spanner<vertex, weight>(G, k);
    t.next("weighted_spanner");
  }

  std::cout << "Spanner size: " << result.size() << " edges" << std::endl;
  std::cout << "Sparsity: " << (double)result.size() / (10 * n) << std::endl;

  return 0;
}
