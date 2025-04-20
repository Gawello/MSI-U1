#include "classifier.h"
#include <iostream>
#include <cmath>

// Pomocnicze funkcje do macierzy
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Matrix transpose(const Matrix& A) {
    int rows = A.size();
    int cols = A[0].size();
    Matrix result(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[j][i] = A[i][j];
    return result;
}
Matrix multiply(const Matrix& A, const Matrix& B) {
    int rows = A.size();
    int cols = B[0].size();
    int inner = B.size();
    Matrix result(rows, std::vector<double>(cols, 0.0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            for (size_t k = 0; k < inner; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}
Vector multiply(const Matrix& A, const Vector& v) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[i] += A[i][j] * v[j];
    return result;
}
Matrix inverse(const Matrix& input) {
    int n = input.size();
    Matrix A = input; // robimy kopię
    Matrix I(n, std::vector<double>(n, 0.0));

    // Tworzymy macierz jednostkową
    for (size_t i = 0; i < n; ++i)
        I[i][i] = 1.0;

    for (size_t i = 0; i < n; ++i) {
        // Szukamy maksimum w kolumnie i (dla stabilności numerycznej)
        double maxEl = std::abs(A[i][i]);
        int maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > maxEl) {
                maxEl = std::abs(A[k][i]);
                maxRow = k;
            }
        }

        // Zamieniamy wiersze (w A i I)
        std::swap(A[i], A[maxRow]);
        std::swap(I[i], I[maxRow]);

        // Dzielimy cały wiersz, aby A[i][i] = 1
        double pivot = A[i][i];
        if (std::abs(pivot) < 1e-12) {
            std::cerr << "Macierz jest osobliwa (nieodwracalna)!\n";
            return Matrix(); // zwracamy pustą macierz
        }

        for (size_t j = 0; j < n; ++j) {
            A[i][j] /= pivot;
            I[i][j] /= pivot;
        }

        // Zerujemy pozostałe elementy w kolumnie i
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                double factor = A[k][i];
                for (size_t j = 0; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }
    }

    return I;
}

std::vector<double> transformFeatures(double x1, double x2) {
    return {
        1.0,
        x1,
        x2,
        x1 * x1,
        x2 * x2,
        x1 * x2
    };
}

void Classifier::train(const std::vector<Vector2D>& samples, const std::vector<int>& labels, int targetClass) {
    Matrix A;
    Vector y;

    for (size_t i = 0; i < samples.size(); ++i) {
        A.push_back(transformFeatures(samples[i].x1, samples[i].x2));
        y.push_back(labels[i] == targetClass ? 1.0 : -1.0);
    }

    Matrix At = transpose(A);
    Matrix AtA = multiply(At, A);

    // Regularyzacja (naprawia osobliwość)
    for (size_t i = 0; i < AtA.size(); ++i)
        AtA[i][i] += 1e-8;

    Matrix AtA_inv = inverse(AtA);
    Vector Aty = multiply(At, y);
    weights = multiply(AtA_inv, Aty);
}

double Classifier::computePotential(const Vector2D& sample) const {
    auto phi = transformFeatures(sample.x1, sample.x2);
    double result = 0.0;
    for (size_t i = 0; i < weights.size(); ++i)
        result += weights[i] * phi[i];
    return result;
}

int Classifier::predictClass(const std::vector<Classifier>& classifiers, const Vector2D& sample) {
    double bestScore = -1e9;
    int bestClass = -1;

    for (size_t k = 0; k < classifiers.size(); ++k) {
        double score = classifiers[k].computePotential(sample);
        if (score > bestScore) {
            bestScore = score;
            bestClass = k;
        }
    }

    return bestClass;
}


const std::vector<double>& Classifier::getWeights() const {
    return weights;
}
