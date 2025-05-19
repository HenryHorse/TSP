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

int findNearestCluster(int clusterID, const std::unordered_map<int, std::vector<int>>& clusters, const std::set<int>& active, ANNpointArray pts) {
    double bestDistance = std::numeric_limits<double>::infinity();
    int bestClusterID = -1;
    const auto &path = clusters.at(clusterID);
    int a0 = path.front(), a1 = path.back();

    // Linearly check every other cluster
    for (int other : active) {
        if (other == clusterID) {
            continue;
        }
        const auto& otherPath = clusters.at(other);
        int b0 = otherPath.front(), b1 = otherPath.back();
        // Find the minimum distance between either endpoint of the given cluster and the endpoints of any other cluster
        for (int clusterAEndpoint : {a0, a1}) {
            for (int clusterBEndpoint : {b0, b1}) {
                double distance = euclidean(pts[clusterAEndpoint], pts[clusterBEndpoint]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClusterID = other;
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

std::vector<std::pair<int, int>> multifragment(ANNpointArray pts, int numPts) {
    std::unordered_map<int, std::vector<int>> clusters;
    std::set<int> active;
    for (int i = 0; i < numPts; i++) {
        clusters[i] = { i };
        active.insert(i);
    }
    std::vector<std::pair<int, int>> edges;
    int nextId = numPts;

    while (active.size() > 1) {
        std::vector<int> stack;
        stack.push_back(*active.begin());

        while (true) {
            int top = stack.back();
            int nearestNeighbor = findNearestCluster(top, clusters, active, pts);
            // If the nearest neighbor is not on the stack, push it onto the stack
            if (std::find(stack.begin(), stack.end(), nearestNeighbor) == stack.end()) {
                stack.push_back(nearestNeighbor);
            }
            else {
                auto merged = mergeClusters(top, nearestNeighbor, clusters, pts, edges);
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
    return edges;
}

int main() {
    int seed = 1337;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0, 10);
    int numPts = 8;
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


    auto tourEdges = multifragment(pts, numPts);

    std::cout << "Multifragment tour edges:\n";
    for (auto& e : tourEdges) {
        std::cout << e.first << " - " << e.second << "\n";
    }
    annDeallocPts(pts);
    return 0;
}