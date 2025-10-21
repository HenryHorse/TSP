//
// Created by Shayan Daijavad on 5/18/25.
//

#ifndef TSP_MULTIFRAGMENT_H
#define TSP_MULTIFRAGMENT_H

#include <ANN/ANN.h>
#include <vector>
#include <utility>
#include <set>
#include <unordered_map>

// Structure to represent the result of an SNN query
struct SNNResult {
    bool isHardAnswer;  // true if we found the true NN, false if soft answer
    int nearestNeighborIdx;  // index of nearest neighbor (if hard answer)
    int softPair1;  // first point of closer pair (if soft answer)
    int softPair2;  // second point of closer pair (if soft answer)
};

// SNN Query using k-ANN Data Structure (Algorithm 1 from paper)
// Given query point q, returns either:
//   - The nearest neighbor of q in P (hard answer)
//   - A pair of points p, p' in P satisfying d(p, p') < d(q, p*) (soft answer)
SNNResult snnQuery(
    ANNpoint q,              // query point
    ANNkd_tree* kdTree,      // k-d tree containing point set P
    int k = 3,               // number of neighbors to retrieve (default 3)
    double eps = 0.0         // error bound (0 = exact)
);

// Euclidean distance between two points
double euclidean(const ANNpoint& a, const ANNpoint& b);

// Find nearest cluster using SNN query
int findNearestCluster(
    int clusterID,
    const std::unordered_map<int, std::vector<int>>& clusters,
    const std::set<int>& active,
    ANNpointArray pts,
    ANNkd_tree* kdTree,
    const std::unordered_map<int, int>& pointToCluster,  // maps point index to cluster ID
    int k = 3,              // number of neighbors for SNN query
    double eps = 0.0        // approximation factor for k-NN
);

// Merge two clusters
std::vector<int> mergeClusters(
    int a,
    int b,
    std::unordered_map<int, std::vector<int>>& clusters,
    ANNpointArray pts,
    std::vector<std::pair<int, int>>& edges
);

// Main multifragment algorithm
// k: number of neighbors for SNN query (default 3)
// eps: approximation factor for k-NN (0.0 = exact, >0 = approximate)
std::vector<std::pair<int, int>> multifragment(ANNpointArray pts, int numPts, int k = 3, double eps = 0.0);


#endif //TSP_MULTIFRAGMENT_H
