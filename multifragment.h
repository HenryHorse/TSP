//
// Created by Shayan Daijavad on 5/18/25.
//

#ifndef TSP_MULTIFRAGMENT_H
#define TSP_MULTIFRAGMENT_H

#include <ANN/ANN.h>
#include <vector>
#include <list>
#include <utility>
#include <set>
#include <unordered_map>

// Structure to represent the result of an SNN query
struct SNNResult {
    bool isHardAnswer;  // true if we found the true NN, false if soft answer
    int nearestNeighborIdx;  // index of nearest neighbor (if hard answer)
    int softPair1;  // first point of closer triple (if soft answer)
    int softPair2;  // second point of closer triple (if soft answer)
    int softPair3;  // third point of closer triple (if soft answer)
};

// SNN Query using k-ANN Data Structure (Algorithm 1 from paper)
// Given query point q, returns either:
//   - The nearest neighbor of q in P (hard answer)
//   - A triple of points p, p', p'' in P that are mutually closer than d(q, p*) (soft answer)
SNNResult snnQuery(
    ANNpoint q,              // query point
    ANNkd_tree* kdTree,      // k-d tree containing point set P
    int k,                   // number of neighbors to retrieve
    double eps               // error bound (0 = exact)
);

// Euclidean distance between two points
double euclidean(const ANNpoint& a, const ANNpoint& b);

// Find nearest cluster using SNN query
int findNearestCluster(
    int clusterID,
    const std::unordered_map<int, std::list<int>>& clusters,
    const std::set<int>& active,
    ANNpointArray pts,
    ANNkd_tree* kdTree,
    const std::unordered_map<int, int>& pointToCluster,  // maps point index to cluster ID
    int k,                  // number of neighbors for SNN query
    double eps              // approximation factor for k-NN
);

// Merge two clusters
std::list<int> mergeClusters(
    int a,
    int b,
    std::unordered_map<int, std::list<int>>& clusters,
    ANNpointArray pts,
    std::vector<std::pair<int, int>>& edges,
    std::unordered_map<int, int>& pointToCluster,
    int newClusterID
);

// Main multifragment algorithm
// k: number of neighbors for SNN query
// eps: approximation factor for k-NN (0.0 = exact, >0 = approximate)
std::vector<std::pair<int, int>> multifragment(ANNpointArray pts, int numPts, int k, double eps);


#endif //TSP_MULTIFRAGMENT_H
