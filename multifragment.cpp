//
// Created by Shayan Daijavad on 5/18/25.
//

#include <ANN/ANN.h>
#include <vector>
#include <list>
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
static const int NULL_CLUSTER = -1;  // Sentinel value for intermediate points in a cluster

double euclidean(const ANNpoint& a, const ANNpoint& b) {
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Implementation of Algorithm 1: SNN Query using k-ANN Data Structure
// This implements the three-way Soft Nearest-Neighbor query from the paper
SNNResult snnQuery(ANNpoint q, ANNkd_tree* kdTree, int k, double epsilon) {
    SNNResult result;

    // Allocate arrays for k-NN search results
    ANNidxArray nn_idx = new ANNidx[k];
    ANNdistArray dists = new ANNdist[k];

    // Call k-ANN to get k nearest neighbors
    kdTree->annkSearch(q, k, nn_idx, dists, epsilon);

    // Distance to true nearest neighbor (first result)
    double d_q_pstar = std::sqrt(dists[0]);

    // Find three mutually close points (a triple where all pairwise distances < d_q_pstar)
    ANNpointArray pts = kdTree->thePoints();

    bool foundTriple = false;
    int tripleIdx1 = -1, tripleIdx2 = -1, tripleIdx3 = -1;

    // Check all triples among p1...pk
    for (int i = 0; i < k && !foundTriple; i++) {
        for (int j = i + 1; j < k && !foundTriple; j++) {
            double dist_ij = euclidean(pts[nn_idx[i]], pts[nn_idx[j]]);
            if (dist_ij >= d_q_pstar) continue;

            for (int l = j + 1; l < k; l++) {
                double dist_il = euclidean(pts[nn_idx[i]], pts[nn_idx[l]]);
                double dist_jl = euclidean(pts[nn_idx[j]], pts[nn_idx[l]]);

                // Check if all three pairwise distances are < d_q_pstar
                if (dist_il < d_q_pstar && dist_jl < d_q_pstar) {
                    foundTriple = true;
                    tripleIdx1 = nn_idx[i];
                    tripleIdx2 = nn_idx[j];
                    tripleIdx3 = nn_idx[l];
                    break;
                }
            }
        }
    }

    // Decide if this is a hard or soft answer
    if (foundTriple) {
        // Soft answer: return the mutually close triple
        result.isHardAnswer = false;
        result.softPair1 = tripleIdx1;
        result.softPair2 = tripleIdx2;
        result.softPair3 = tripleIdx3;
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

// Helper: Compute minimum distance between endpoints of two clusters
double distanceBetweenClusters(
    const std::list<int>& pathA,
    const std::list<int>& pathB,
    ANNpointArray pts
) {
    int a0 = pathA.front(), a1 = pathA.back();
    int b0 = pathB.front(), b1 = pathB.back();

    double minDist = std::numeric_limits<double>::infinity();
    for (int endpointA : {a0, a1}) {
        for (int endpointB : {b0, b1}) {
            double dist = euclidean(pts[endpointA], pts[endpointB]);
            if (dist < minDist) {
                minDist = dist;
            }
        }
    }
    return minDist;
}

// Helper: Map a point index to its cluster and validate it's a viable candidate
// Returns -1 if point is invalid, same cluster, or not active
int getValidCandidateCluster(
    int pointIdx,
    int myClusterID,
    const std::unordered_map<int, int>& pointToCluster,
    const std::unordered_map<int, std::list<int>>& clusters,
    const std::set<int>& active
) {
    // Validate point index
    if (pointIdx == -1) return -1;

    // Map point to its cluster
    auto it = pointToCluster.find(pointIdx);
    if (it == pointToCluster.end()) return -1;

    int candidateCluster = it->second;

    // Filter out invalid candidates
    if (candidateCluster == NULL_CLUSTER) return -1;          // Intermediate point (not an endpoint)
    if (candidateCluster == myClusterID) return -1;           // Same cluster
    if (active.count(candidateCluster) == 0) return -1;       // Not active
    if (clusters.find(candidateCluster) == clusters.end()) return -1;  // Doesn't exist

    return candidateCluster;
}

// Helper: Update best cluster if candidate is better
void updateBestCluster(
    int candidateCluster,
    const std::list<int>& myPath,
    const std::unordered_map<int, std::list<int>>& clusters,
    ANNpointArray pts,
    int& bestClusterID,
    double& bestDistance
) {
    if (candidateCluster == -1) return;

    const auto& candidatePath = clusters.at(candidateCluster);
    double dist = distanceBetweenClusters(myPath, candidatePath, pts);

    if (dist < bestDistance) {
        bestDistance = dist;
        bestClusterID = candidateCluster;
    }
}

// Modified version using SNN query for finding nearest cluster
// This queries from the cluster endpoints using the k-d tree
int findNearestCluster(
    int clusterID,
    const std::unordered_map<int, std::list<int>>& clusters,
    const std::set<int>& active,
    ANNpointArray pts,
    ANNkd_tree* kdTree,
    const std::unordered_map<int, int>& pointToCluster,
    int k,
    double eps
) {
    const auto &path = clusters.at(clusterID);
    int a0 = path.front(), a1 = path.back();

    // Query from both endpoints of the cluster
    SNNResult result1 = snnQuery(pts[a0], kdTree, k, eps);
    SNNResult result2 = (a0 != a1) ? snnQuery(pts[a1], kdTree, k, eps) : result1;

    double bestDistance = std::numeric_limits<double>::infinity();
    int bestClusterID = -1;

    // Algorithm 2: Process results based on hard/soft answers from both endpoints
    if (result1.isHardAnswer && result2.isHardAnswer) {
        // Both endpoints got hard answers - check both candidates
        int candidate1 = getValidCandidateCluster(result1.nearestNeighborIdx, clusterID, pointToCluster, clusters, active);
        int candidate2 = getValidCandidateCluster(result2.nearestNeighborIdx, clusterID, pointToCluster, clusters, active);
        updateBestCluster(candidate1, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate2, path, clusters, pts, bestClusterID, bestDistance);
    }
    else if (!result1.isHardAnswer && !result2.isHardAnswer) {
        // Both got soft answers - check all six points from both triples
        int candidate1 = getValidCandidateCluster(result1.softPair1, clusterID, pointToCluster, clusters, active);
        int candidate2 = getValidCandidateCluster(result1.softPair2, clusterID, pointToCluster, clusters, active);
        int candidate3 = getValidCandidateCluster(result1.softPair3, clusterID, pointToCluster, clusters, active);
        int candidate4 = getValidCandidateCluster(result2.softPair1, clusterID, pointToCluster, clusters, active);
        int candidate5 = getValidCandidateCluster(result2.softPair2, clusterID, pointToCluster, clusters, active);
        int candidate6 = getValidCandidateCluster(result2.softPair3, clusterID, pointToCluster, clusters, active);
        updateBestCluster(candidate1, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate2, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate3, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate4, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate5, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate6, path, clusters, pts, bestClusterID, bestDistance);
    }
    else if (!result1.isHardAnswer) {
        // result1 soft, result2 hard - check triple from result1 and hard answer from result2
        int candidate1 = getValidCandidateCluster(result1.softPair1, clusterID, pointToCluster, clusters, active);
        int candidate2 = getValidCandidateCluster(result1.softPair2, clusterID, pointToCluster, clusters, active);
        int candidate3 = getValidCandidateCluster(result1.softPair3, clusterID, pointToCluster, clusters, active);
        int candidate4 = getValidCandidateCluster(result2.nearestNeighborIdx, clusterID, pointToCluster, clusters, active);
        updateBestCluster(candidate1, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate2, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate3, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate4, path, clusters, pts, bestClusterID, bestDistance);
    }
    else {
        // result1 hard, result2 soft - check hard answer from result1 and triple from result2
        int candidate1 = getValidCandidateCluster(result1.nearestNeighborIdx, clusterID, pointToCluster, clusters, active);
        int candidate2 = getValidCandidateCluster(result2.softPair1, clusterID, pointToCluster, clusters, active);
        int candidate3 = getValidCandidateCluster(result2.softPair2, clusterID, pointToCluster, clusters, active);
        int candidate4 = getValidCandidateCluster(result2.softPair3, clusterID, pointToCluster, clusters, active);
        updateBestCluster(candidate1, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate2, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate3, path, clusters, pts, bestClusterID, bestDistance);
        updateBestCluster(candidate4, path, clusters, pts, bestClusterID, bestDistance);
    }

    // Fallback: if SNN didn't find a valid cluster, do linear search
    if (bestClusterID == -1) {
        for (int otherCluster : active) {
            if (otherCluster == clusterID) continue;
            const auto& otherPath = clusters.at(otherCluster);
            double dist = distanceBetweenClusters(path, otherPath, pts);
            if (dist < bestDistance) {
                bestDistance = dist;
                bestClusterID = otherCluster;
            }
        }
    }

    return bestClusterID;
}

std::list<int> mergeClusters(int a, int b, std::unordered_map<int, std::list<int>>& clusters,
                             ANNpointArray pts, std::vector<std::pair<int, int>>& edges,
                             std::unordered_map<int, int>& pointToCluster, int newClusterID) {
    auto& pathA = clusters[a];
    auto& pathB = clusters[b];

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

    // Reverse paths if needed so that the connection is at the ends
    if (pathA.front() == bestAEndpoint) {
        pathA.reverse();
    }

    if (pathB.back() == bestBEndpoint) {
        pathB.reverse();
    }

    // O(1) splice operation to merge the lists
    pathA.splice(pathA.end(), pathB);

    // Mark all intermediate points as NULL_CLUSTER (O(n) but only needs to happen once per point)
    // Only the two endpoints of the merged cluster remain valid for queries
    auto it = pathA.begin();
    while (it != pathA.end()) {
        pointToCluster[*it] = NULL_CLUSTER;
        ++it;
    }

    // Update only the two endpoints to the new cluster ID (O(1))
    pointToCluster[pathA.front()] = newClusterID;
    pointToCluster[pathA.back()] = newClusterID;

    return pathA;
}

std::vector<std::pair<int, int>> multifragment(ANNpointArray pts, int numPts, int k, double eps) {
    std::unordered_map<int, std::list<int>> clusters;
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
        // Stack now stores pairs: (cluster, its nearest neighbor)
        std::vector<std::pair<int, int>> stack;

        // Start with an arbitrary cluster
        int startCluster = *active.begin();
        int startNN = findNearestCluster(startCluster, clusters, active, pts, kdTree, pointToCluster, k, eps);
        stack.push_back({startCluster, startNN});

        while (true) {
            // Get the nearest neighbor from the top pair
            int currentCluster = stack.back().second;
            int currentNN = findNearestCluster(currentCluster, clusters, active, pts, kdTree, pointToCluster, k, eps);

            // Check if we have a mutual nearest neighbor relationship (cycle of length 2)
            // currentCluster's NN is currentNN, check if currentNN's NN is currentCluster
            if (currentNN == stack.back().first) {
                // Found a cycle: currentCluster <-> currentNN (mutual nearest neighbors)
                // Merge currentCluster and currentNN
                auto merged = mergeClusters(currentCluster, currentNN, clusters, pts, edges, pointToCluster, nextId);

                active.erase(currentCluster);
                active.erase(currentNN);
                clusters.erase(currentCluster);
                clusters.erase(currentNN);
                clusters[nextId] = std::move(merged);
                active.insert(nextId);

                // Remove pairs involving the merged clusters from the stack
                // The top pair was (currentNN, currentCluster) and needs to be removed
                stack.pop_back();

                if (!stack.empty()) {
                    // The previous pair (X, currentNN) also references a deleted cluster (currentNN)
                    int prevCluster = stack.back().first;
                    stack.pop_back();  // Remove (X, currentNN)

                    // Continue from prevCluster, requerying its nearest neighbor
                    int newNN = findNearestCluster(prevCluster, clusters, active, pts, kdTree, pointToCluster, k, eps);
                    stack.push_back({prevCluster, newNN});
                } else {
                    // Stack is empty, start fresh with the merged cluster
                    int newNN = findNearestCluster(nextId, clusters, active, pts, kdTree, pointToCluster, k, eps);
                    stack.push_back({nextId, newNN});
                }

                nextId++;
                break;
            } else {
                // No cycle yet, push the new pair onto the stack
                stack.push_back({currentCluster, currentNN});
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
    std::uniform_real_distribution<> distribution(0, 100);
    int numPts = 100;
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);

    for (int i = 0; i < numPts; i++) {
        pts[i][0] = distribution(gen);
        pts[i][1] = distribution(gen);
    }

    std::cout << "Generated Points:\n";
    for (int i = 0; i < numPts; i++) {
        std::cout << "Point " << i << ": ("
                  << pts[i][0] << ", "
                  << pts[i][1] << ")\n";
    }

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