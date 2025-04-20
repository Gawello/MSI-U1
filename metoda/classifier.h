#pragma once
#include <vector>
#include "../problem/vector2d.h"

class Classifier {
private:
    std::vector<double> weights;

public:
    void train(const std::vector<Vector2D>& samples, const std::vector<int>& labels, int targetClass);
    double computePotential(const Vector2D& sample) const;
    static int predictClass(const std::vector<Classifier>& classifiers, const Vector2D& sample);
    const std::vector<double>& getWeights() const;
};
