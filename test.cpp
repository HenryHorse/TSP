//
// Created by Shayan Daijavad on 6/6/25.
//

#include "test.h"
#include "multifragment.h"
#include <gtest/gtest.h>
#include <ANN/ANN.h>
#include <random>
#include <cmath>
#include <unordered_map>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

// Test case where p1 and p2 are on the same path/cluster
// This tests the edge case in snnQuery where the k nearest neighbors
// might include points that are closer to each other than to the query point
TEST(SNNQueryTest, PointsOnSamePath) {
    const int DIMENSIONS = 2;
    const int numPts = 6;

    // Create a simple point set where some points are very close to each other
    // Point 0: (0, 0) - will be used to create query point (not in tree)
    // Point 1: (10, 0) - relatively far
    // Point 2: (10.1, 0) - very close to point 1 (same "path")
    // Point 3: (10.2, 0) - also very close to point 1 (same "path")
    // Point 4: (20, 0) - far away
    // Point 5: (30, 0) - very far
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);
    pts[0][0] = 0;   pts[0][1] = 0;
    pts[1][0] = 10;  pts[1][1] = 0;
    pts[2][0] = 10.1; pts[2][1] = 0;
    pts[3][0] = 10.2; pts[3][1] = 0;
    pts[4][0] = 20;  pts[4][1] = 0;
    pts[5][0] = 30;  pts[5][1] = 0;

    // Build k-d tree with all points
    ANNkd_tree* kdTree = new ANNkd_tree(pts, numPts, DIMENSIONS);

    // Create a separate query point not in the tree
    ANNpoint queryPt = annAllocPt(DIMENSIONS);
    queryPt[0] = 0.5;
    queryPt[1] = 0;

    // Query with k=4 - should get points 0, 1, 2, 3
    // Point 0 is closest (~0.5), points 1,2,3 are around distance ~9.5-9.7
    SNNResult result = snnQuery(queryPt, kdTree, 4, 0.0);

    // Expected: Should return soft answer because points 1, 2, 3 are closer
    // to each other (distance ~0.1-0.2) than query is to nearest neighbor
    // Actually, since point 0 at (0,0) is closest and distance is 0.5,
    // and points 1,2,3 have pairwise distances 0.1-0.2, we should get soft answer
    EXPECT_FALSE(result.isHardAnswer);

    // The soft pair should be points that are very close to each other
    // Most likely (1,2), (1,3), or (2,3)
    EXPECT_TRUE(
        (result.softPair1 == 1 && result.softPair2 == 2) ||
        (result.softPair1 == 1 && result.softPair2 == 3) ||
        (result.softPair1 == 2 && result.softPair2 == 3) ||
        (result.softPair1 == 2 && result.softPair2 == 1) ||
        (result.softPair1 == 3 && result.softPair2 == 1) ||
        (result.softPair1 == 3 && result.softPair2 == 2) ||
        // Could also be point 0 paired with another
        (result.softPair1 == 0 && result.softPair2 == 1) ||
        (result.softPair1 == 1 && result.softPair2 == 0)
    );

    // Clean up
    delete kdTree;
    annDeallocPt(queryPt);
    annDeallocPts(pts);
}

// Test case for hard answer from snnQuery
TEST(SNNQueryTest, HardAnswer) {
    const int DIMENSIONS = 2;
    const int numPts = 4;

    // Create points where no pair is closer than query to nearest neighbor
    // Point 0: (0, 0)
    // Point 1: (10, 0)
    // Point 2: (20, 0)
    // Point 3: (30, 0)
    // All points are evenly spaced with distance 10
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);
    pts[0][0] = 0;  pts[0][1] = 0;
    pts[1][0] = 10; pts[1][1] = 0;
    pts[2][0] = 20; pts[2][1] = 0;
    pts[3][0] = 30; pts[3][1] = 0;

    ANNkd_tree* kdTree = new ANNkd_tree(pts, numPts, DIMENSIONS);

    // Create a query point separate from the dataset
    ANNpoint queryPt = annAllocPt(DIMENSIONS);
    queryPt[0] = 5;  // Halfway between point 0 and 1
    queryPt[1] = 0;

    // Query with k=3 - will get points 1 (d=5), 0 (d=5), 2 (d=15)
    SNNResult result = snnQuery(queryPt, kdTree, 3, 0.0);

    // The nearest neighbor is either point 0 or 1 (both at distance 5)
    // The minimum pairwise distance among the k-NN is d(0,1)=10
    // Since 10 > 5, this should be a hard answer
    EXPECT_TRUE(result.isHardAnswer);
    // The nearest neighbor should be either 0 or 1
    EXPECT_TRUE(result.nearestNeighborIdx == 0 || result.nearestNeighborIdx == 1);

    // Clean up
    delete kdTree;
    annDeallocPt(queryPt);
    annDeallocPts(pts);
}

// Test multifragment on a small simple case
TEST(MultifragmentTest, SmallTour) {
    const int DIMENSIONS = 2;
    const int numPts = 4;

    // Create a square of points
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);
    pts[0][0] = 0; pts[0][1] = 0;
    pts[1][0] = 1; pts[1][1] = 0;
    pts[2][0] = 1; pts[2][1] = 1;
    pts[3][0] = 0; pts[3][1] = 1;

    auto edges = multifragment(pts, numPts);

    // Should have exactly numPts edges (forming a cycle)
    EXPECT_EQ(edges.size(), numPts);

    // Clean up
    annDeallocPts(pts);
}

// Test multifragment with approximate queries (k=20, epsilon as specified)
TEST(MultifragmentTest, ApproximateQueries) {
    const int DIMENSIONS = 2;
    const int numPts = 50;

    // Create random points for testing
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0, 100);
    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);

    for (int i = 0; i < numPts; i++) {
        pts[i][0] = dist(gen);
        pts[i][1] = dist(gen);
    }

    // Test with specified k and epsilon from TODO
    int k = 20;
    double epsilon = std::pow((1.0 + std::sqrt(5.0)) / 2, 1.0 / 20.0) - 1.0;

    auto edges = multifragment(pts, numPts, k, epsilon);

    // Should have exactly numPts edges (forming a cycle)
    EXPECT_EQ(edges.size(), numPts);

    // Verify it's a valid tour (each point appears exactly twice)
    std::unordered_map<int, int> degree;
    for (const auto& edge : edges) {
        degree[edge.first]++;
        degree[edge.second]++;
    }

    for (int i = 0; i < numPts; i++) {
        EXPECT_EQ(degree[i], 2) << "Point " << i << " should have degree 2";
    }

    // Clean up
    annDeallocPts(pts);
}

// Test that different k and epsilon values produce valid tours
TEST(MultifragmentTest, ParameterVariation) {
    const int DIMENSIONS = 2;
    const int numPts = 20;

    ANNpointArray pts = annAllocPts(numPts, DIMENSIONS);
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dist(0, 50);

    for (int i = 0; i < numPts; i++) {
        pts[i][0] = dist(gen);
        pts[i][1] = dist(gen);
    }

    // Test with k=3 (small), exact
    auto edges1 = multifragment(pts, numPts, 3, 0.0);
    EXPECT_EQ(edges1.size(), numPts);

    // Test with k=10, approximate
    auto edges2 = multifragment(pts, numPts, 10, 0.1);
    EXPECT_EQ(edges2.size(), numPts);

    // Test with k=20
    int k = 20;
    double epsilon = std::pow((1.0 + std::sqrt(5.0)) / 2, 1.0 / 20.0) - 1.0;
    auto edges3 = multifragment(pts, numPts, k, epsilon);
    EXPECT_EQ(edges3.size(), numPts);

    // Clean up
    annDeallocPts(pts);
}

