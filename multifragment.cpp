//
// Created by Shayan Daijavad on 5/18/25.
//

#include <ANN/ANN.h>
#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <limits>
#include <utility>
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include "multifragment.h"


static const int DIMENSIONS = 2;

double euclidean(const ANNpoint& a, const ANNpoint& b) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Implementation of Algorithm 1: SNN Query using k-ANN Data Structure
// This implements the Soft Nearest-Neighbor query from the paper
SNNResult snnQuery(ANNpoint q, ANNkd_tree* kdTree, int k, double eps) {
    SNNResult result;

    // Allocate arrays for k-NN search results
    ANNidxArray nn_idx = new ANNidx[k];
    ANNdistArray dists = new ANNdist[k];

    // Call k-ANN to get k nearest neighbors
    kdTree->annkSearch(q, k, nn_idx, dists, eps);

    // Distance to true nearest neighbor (first result)
    double d_q_pstar = std::sqrt(dists[0]);

    // Check if any pair among p1...pk is closer to each other than q is to p*
    double minPairDist = std::numeric_limits<double>::infinity();
    int minPairIdx1 = -1, minPairIdx2 = -1;

    ANNpointArray pts = kdTree->thePoints();

    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            double dist_ij = euclidean(pts[nn_idx[i]], pts[nn_idx[j]]);
            if (dist_ij < minPairDist) {
                minPairDist = dist_ij;
                minPairIdx1 = nn_idx[i];
                minPairIdx2 = nn_idx[j];
            }
        }
    }

    // Decide if this is a hard or soft answer
    if (minPairDist < d_q_pstar) {
        // Soft answer: return the closer pair
        result.isHardAnswer = false;
        result.softPair1 = minPairIdx1;
        result.softPair2 = minPairIdx2;
    } else {
        // Hard answer: return the true nearest neighbor
        result.isHardAnswer = true;
        result.nearestNeighborIdx = nn_idx[0];
    }

    // Clean up
    delete[] nn_idx;
    delete[] dists;

    return result;
}

// Modified version using SNN query for finding nearest cluster
// This queries from the cluster endpoints using the k-d tree
int findNearestCluster(
    int clusterID,
    const std::unordered_map<int, std::vector<int>>& clusters,
    const std::set<int>& active,
    ANNpointArray pts,
    ANNkd_tree* kdTree,
    const std::unordered_map<int, int>& pointToCluster,
    int k,
    double eps
) {
    const auto &path = clusters.at(clusterID);
    int a0 = path.front(), a1 = path.back();

    // Query from both endpoints of the cluster with configurable k and epsilon
    SNNResult result1 = snnQuery(pts[a0], kdTree, k, eps);
    SNNResult result2 = (a0 != a1) ? snnQuery(pts[a1], kdTree, k, eps) : result1;

    double bestDistance = std::numeric_limits<double>::infinity();
    int bestClusterID = -1;

    // Helper function to check distance and update best
    auto checkCandidate = [&](int pointIdx) {
        // Check if point exists in mapping
        auto it = pointToCluster.find(pointIdx);
        if (it == pointToCluster.end()) return;

        int candidateCluster = it->second;

        // Skip if it's the same cluster or not active
        if (candidateCluster == clusterID) return;
        if (active.count(candidateCluster) == 0) return;

        // Check if cluster still exists
        auto clusterIt = clusters.find(candidateCluster);
        if (clusterIt == clusters.end()) return;

        const auto& otherPath = clusterIt->second;
        int b0 = otherPath.front(), b1 = otherPath.back();

        // Check distances between endpoints
        for (int myEndpoint : {a0, a1}) {
            for (int otherEndpoint : {b0, b1}) {
                double distance = euclidean(pts[myEndpoint], pts[otherEndpoint]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClusterID = candidateCluster;
                }
            }
        }
    };

    // Process hard answers
    if (result1.isHardAnswer) {
        checkCandidate(result1.nearestNeighborIdx);
    } else {
        // Process soft answer - check both points in the pair
        checkCandidate(result1.softPair1);
        checkCandidate(result1.softPair2);
    }

    if (result2.isHardAnswer) {
        checkCandidate(result2.nearestNeighborIdx);
    } else {
        checkCandidate(result2.softPair1);
        checkCandidate(result2.softPair2);
    }

    // Fallback: if SNN didn't find a valid cluster, do linear search
    // This can happen when k is too small or all nearby points are in the same cluster
    if (bestClusterID == -1) {
        for (int other : active) {
            if (other == clusterID) continue;

            const auto& otherPath = clusters.at(other);
            int b0 = otherPath.front(), b1 = otherPath.back();

            for (int myEndpoint : {a0, a1}) {
                for (int otherEndpoint : {b0, b1}) {
                    double distance = euclidean(pts[myEndpoint], pts[otherEndpoint]);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestClusterID = other;
                    }
                }
            }
        }
    }

    return bestClusterID;
}

std::vector<int> mergeClusters(int a, int b, std::unordered_map<int, std::vector<int>>& clusters, ANNpointArray pts, std::vector<std::pair<int, int>>& edges) {
    auto pathA = clusters[a];
    auto pathB = clusters[b];

    int bestAEndpoint = -1;
    int bestBEndpoint = -1;
    double bestDist = std::numeric_limits<double>::infinity();

    std::vector<std::pair<int, int>> endpointPairs = {
            {pathA.front(), pathB.front()},
            {pathA.front(), pathB.back()},
            {pathA.back(), pathB.front()},
            {pathA.back(), pathB.back()}
    };
    for (auto& pair: endpointPairs) {
        double distance = euclidean(pts[pair.first], pts[pair.second]);
        if (distance < bestDist) {
            bestDist = distance;
            bestAEndpoint = pair.first;
            bestBEndpoint = pair.second;
        }
    }

    edges.emplace_back(bestAEndpoint, bestBEndpoint);

    if (pathA.front() == bestAEndpoint) {
        std::reverse(pathA.begin(), pathA.end());
    }

    if (pathB.back() == bestBEndpoint) {
        std::reverse(pathB.begin(), pathB.end());
    }

    pathA.insert(pathA.end(), pathB.begin(), pathB.end());

    return pathA;
}

std::vector<std::pair<int, int>> multifragment(ANNpointArray pts, int numPts, int k, double eps) {
    std::unordered_map<int, std::vector<int>> clusters;
    std::set<int> active;
    std::unordered_map<int, int> pointToCluster;  // maps point index to cluster ID

    // Initialize: each point is its own cluster
    for (int i = 0; i < numPts; i++) {
        clusters[i] = { i };
        active.insert(i);
        pointToCluster[i] = i;
    }

    std::vector<std::pair<int, int>> edges;
    int nextId = numPts;

    // Build initial k-d tree with all points
    ANNkd_tree* kdTree = new ANNkd_tree(pts, numPts, DIMENSIONS);

    while (active.size() > 1) {
        std::vector<int> stack;
        stack.push_back(*active.begin());

        while (true) {
            int top = stack.back();
            int nearestNeighbor = findNearestCluster(top, clusters, active, pts, kdTree, pointToCluster, k, eps);

            // If the nearest neighbor is not on the stack, push it onto the stack
            if (std::find(stack.begin(), stack.end(), nearestNeighbor) == stack.end()) {
                stack.push_back(nearestNeighbor);
            }
            else {
                // Merge the two clusters
                auto merged = mergeClusters(top, nearestNeighbor, clusters, pts, edges);

                // Update pointToCluster mapping for the merged cluster
                for (int pointIdx : merged) {
                    pointToCluster[pointIdx] = nextId;
                }

                active.erase(top);
                active.erase(nearestNeighbor);
                clusters.erase(top);
                clusters.erase(nearestNeighbor);
                clusters[nextId] = std::move(merged);
                active.insert(nextId);
                stack.clear();
                stack.push_back(nextId);
                nextId++;
                break;
            }
        }
    }

    // Close the cycle
    const auto& finalPath = clusters.begin()->second;
    edges.emplace_back(finalPath.back(), finalPath.front());

    // Clean up
    delete kdTree;

    return edges;
}

#ifndef TESTING
int main() {
    int seed = 1339;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0, 100);
    int numPts = 200;
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);

    for (int i = 0; i < numPts; i++) {
        pts[i][0] = dist(gen);
        pts[i][1] = dist(gen);
    }

    std::cout << "Generated Points:\n";
    for (int i = 0; i < numPts; i++) {
        std::cout << "Point " << i << ": ("
                  << pts[i][0] << ", "
                  << pts[i][1] << ")\n";
    }

    // Configure SNN query parameters as per TODO:
    // k = 20
    // epsilon = ((1 + sqrt(5)) / 20)^(1/20) - 1
    int k = 20;
    double epsilon = std::pow((1.0 + std::sqrt(5.0)) / 2, 1.0 / 20.0) - 1.0;

    std::cout << "\nRunning multifragment with k=" << k
              << ", epsilon=" << epsilon << "\n\n";

    auto tourEdges = multifragment(pts, numPts, k, epsilon);

    std::cout << "Multifragment tour edges:\n";
    for (auto& e : tourEdges) {
        std::cout << e.first << " - " << e.second << "\n";
    }



    std::ofstream out("output.txt");
    out << numPts << "\n";
    for (int i = 0; i < numPts; i++) {
        out << pts[i][0] << " " << pts[i][1] << "\n";
    }

    out << tourEdges.size() << "\n";
    for (auto& e : tourEdges) {
        out << e.first << " " << e.second << "\n";
    }
    out.close();

    annDeallocPts(pts);
    return 0;
}
#endif