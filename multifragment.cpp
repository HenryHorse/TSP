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
#include <chrono>
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
SNNResult snnQuery(ANNpoint q, ANNbd_tree* bdTree, int k, double epsilon) {
    SNNResult result;

    // Cap k at the number of non-deleted points
    int non_deleted = bdTree->nPoints() - bdTree->nDeleted();
    int actual_k = std::min(k, non_deleted);

    if (actual_k <= 0) {
        result.isHardAnswer = false;
        return result;
    }

    // Allocate arrays for k-NN search results
    ANNidxArray nn_idx = new ANNidx[actual_k];
    ANNdistArray dists = new ANNdist[actual_k];

    // Call k-ANN to get k nearest neighbors
    bdTree->annkSearch(q, actual_k, nn_idx, dists, epsilon);

    // Distance to true nearest neighbor (first result)
    double d_q_pstar = std::sqrt(dists[0]);

    // Find three mutually close points (a triple where all pairwise distances < d_q_pstar)
    ANNpointArray pts = bdTree->thePoints();

    bool foundTriple = false;
    int tripleIdx1 = -1, tripleIdx2 = -1, tripleIdx3 = -1;

    // Check all triples among p1...p(actual_k) - NOTE: using actual_k not k!
    for (int i = 0; i < actual_k && !foundTriple; i++) {
        for (int j = i + 1; j < actual_k && !foundTriple; j++) {
            double dist_ij = euclidean(pts[nn_idx[i]], pts[nn_idx[j]]);
            if (dist_ij >= d_q_pstar) continue;

            for (int l = j + 1; l < actual_k; l++) {
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

// Implementation of SNN query for paths (Algorithm 2 from paper)
// This implements the path-level SNN query using point-level SNN queries
int findNearestCluster(
    int clusterID,
    const std::unordered_map<int, std::list<int>>& clusters,
    const std::set<int>& active,
    ANNpointArray pts,
    ANNbd_tree* bdTree,
    const std::unordered_map<int, int>& pointToCluster,
    int k,
    double eps
) {
    const auto &path = clusters.at(clusterID);
    int q1 = path.front(), q2 = path.back();

    // Temporarily delete our own endpoints to avoid self-matches
    bdTree->deletePoint(q1);
    if (q1 != q2) {
        bdTree->deletePoint(q2);
    }

    // Query S with q1 and q2
    SNNResult answer1 = snnQuery(pts[q1], bdTree, k, eps);
    SNNResult answer2;
    if (q1 != q2) {
        answer2 = snnQuery(pts[q2], bdTree, k, eps);
    } else {
        answer2 = answer1;
    }

    // Restore our endpoints
    bdTree->undeletePoint(q1);
    if (q1 != q2) {
        bdTree->undeletePoint(q2);
    }

    // Collect candidate clusters from the SNN results
    std::set<int> candidateClusters;

    // Case 1: Both answers are hard
    if (answer1.isHardAnswer && answer2.isHardAnswer) {
        int p1 = answer1.nearestNeighborIdx;
        int p2 = answer2.nearestNeighborIdx;

        auto it1 = pointToCluster.find(p1);
        auto it2 = pointToCluster.find(p2);

        if (it1 != pointToCluster.end() && it1->second != NULL_CLUSTER && it1->second != clusterID) {
            candidateClusters.insert(it1->second);
        }
        if (it2 != pointToCluster.end() && it2->second != NULL_CLUSTER && it2->second != clusterID) {
            candidateClusters.insert(it2->second);
        }
    }
    // Case 2: One hard, one soft
    else if (answer1.isHardAnswer != answer2.isHardAnswer) {
        // Add cluster from hard answer
        int hardIdx = answer1.isHardAnswer ? answer1.nearestNeighborIdx : answer2.nearestNeighborIdx;
        auto it = pointToCluster.find(hardIdx);
        if (it != pointToCluster.end() && it->second != NULL_CLUSTER && it->second != clusterID) {
            candidateClusters.insert(it->second);
        }

        // Add clusters from soft answer (the triple)
        std::vector<int> softIndices;
        if (answer1.isHardAnswer) {
            softIndices = {answer2.softPair1, answer2.softPair2, answer2.softPair3};
        } else {
            softIndices = {answer1.softPair1, answer1.softPair2, answer1.softPair3};
        }

        for (int idx : softIndices) {
            auto it = pointToCluster.find(idx);
            if (it != pointToCluster.end() && it->second != NULL_CLUSTER && it->second != clusterID) {
                candidateClusters.insert(it->second);
            }
        }
    }
    // Case 3: Both soft
    else {
        // Add all clusters from both triples
        std::vector<int> allIndices = {
            answer1.softPair1, answer1.softPair2, answer1.softPair3,
            answer2.softPair1, answer2.softPair2, answer2.softPair3
        };

        for (int idx : allIndices) {
            auto it = pointToCluster.find(idx);
            if (it != pointToCluster.end() && it->second != NULL_CLUSTER && it->second != clusterID) {
                candidateClusters.insert(it->second);
            }
        }
    }

    // Find the closest cluster among candidates
    int bestClusterID = -1;
    double bestDistance = std::numeric_limits<double>::infinity();

    for (int candidateID : candidateClusters) {
        // Verify cluster still exists
        if (clusters.find(candidateID) == clusters.end()) {
            continue;
        }

        double dist = distanceBetweenClusters(path, clusters.at(candidateID), pts);
        if (dist < bestDistance) {
            bestDistance = dist;
            bestClusterID = candidateID;
        }
    }

    return bestClusterID;
}

std::list<int> mergeClusters(int a, int b, std::unordered_map<int, std::list<int>>& clusters,
                             ANNpointArray pts, std::vector<std::pair<int, int>>& edges,
                             std::unordered_map<int, int>& pointToCluster, int newClusterID,
                             ANNbd_tree* bdTree) {
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

    // Determine the new endpoints after merging
    int newEndpoint0 = pathA.front();
    int newEndpoint1 = pathA.back();

    // Mark all intermediate points as NULL_CLUSTER and delete them from the tree
    // Skip the endpoints - they should remain active
    auto it = pathA.begin();
    while (it != pathA.end()) {
        int pointIdx = *it;
        pointToCluster[pointIdx] = NULL_CLUSTER;

        // Only delete if it's not one of the new endpoints
        if (pointIdx != newEndpoint0 && pointIdx != newEndpoint1) {
            bdTree->deletePoint(pointIdx);
        }
        ++it;
    }

    // Update only the two endpoints to the new cluster ID
    pointToCluster[newEndpoint0] = newClusterID;
    pointToCluster[newEndpoint1] = newClusterID;

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

    // Build initial bd-tree with all points
    ANNbd_tree* bdTree = new ANNbd_tree(pts, numPts, DIMENSIONS);

    int iteration = 0;
    while (active.size() > 1) {
        if (iteration % 10 == 0) {
            std::cout << "Iter " << iteration << ": " << active.size() << " clusters active" << std::endl;
        }
        iteration++;

        // Stack stores pairs of cluster IDs forming edges in the nearest-neighbor chain
        // Each pair (u, v) represents "u's nearest neighbor is v"
        std::vector<std::pair<int, int>> stack;

        // Start with an arbitrary cluster (single node, so pair is (cluster, cluster))
        int currentCluster = *active.begin();

        // Build the nearest-neighbor chain until we find a cycle
        while (true) {
            int nearestCluster = findNearestCluster(currentCluster, clusters, active, pts, bdTree, pointToCluster, k, eps);

            if (nearestCluster == -1) {
                // No valid nearest neighbor found - shouldn't happen but handle gracefully
                std::cerr << "Warning: No nearest neighbor found for cluster " << currentCluster << std::endl;
                break;
            }

            // Check if nearestCluster is already in the stack (found a cycle)
            // We check if nearestCluster appears in any pair in the stack
            bool foundCycle = false;
            for (const auto& pair : stack) {
                if (pair.first == nearestCluster || pair.second == nearestCluster) {
                    foundCycle = true;
                    break;
                }
            }

            if (foundCycle) {
                // Found a cycle! Merge currentCluster and nearestCluster
                auto merged = mergeClusters(currentCluster, nearestCluster, clusters, pts, edges, pointToCluster, nextId, bdTree);

                active.erase(currentCluster);
                active.erase(nearestCluster);
                clusters.erase(currentCluster);
                clusters.erase(nearestCluster);
                clusters[nextId] = std::move(merged);
                active.insert(nextId);

                nextId++;
                break;  // Start a new chain
            } else {
                // Continue the chain: add the edge (currentCluster -> nearestCluster)
                stack.push_back({currentCluster, nearestCluster});
                currentCluster = nearestCluster;
            }
        }
    }

    // Close the tour
    const auto& finalPath = clusters.begin()->second;
    edges.emplace_back(finalPath.back(), finalPath.front());

    // Clean up
    delete bdTree;

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

    std::cout << "\nMultifragment complete!" << std::endl;

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