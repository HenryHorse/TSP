//
// Created by Shayan Daijavad on 4/23/25.
//

#include <iostream>
#include <vector>
#include <random>
#include "random_points.h"

using Point = std::pair<double, double>;

std::vector<Point> generate_random_points(int n, unsigned int seed) {
    std::vector<Point> points;
    points.reserve(n);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0, 1000);

    for (int i = 0; i < n; i++) {
        points.emplace_back(dist(gen), dist(gen));
    }

    return points;
}


void print_points(const std::vector<Point>& points) {
    for (const auto & point : points) {
        std::cout << "(" << point.first << ", " << point.second << ")\n";
    }
}

int main() {
    int n = 10;
    auto points = generate_random_points(n, 1337);
    print_points(points);
    return 0;
}