#pragma once
#include <vector>
#include "vector2d.h"

using Image = std::vector<std::vector<int>>;

inline Image testImage = {
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 0}};

// inline std::vector<Vector2D> samples = {
//     {8, 0.0},
//     {7, -0.1},
//     {6, -0.2},
//     {4, -0.4},
//     {2, -0.6},
//     {1, -0.8}};

// inline std::vector<int> labels = {
//     0, 0, 1, 1, 2, 2};

inline int countSameNeighbors(const Image &img, int x, int y)
{
    int same = 0;
    int h = img.size(), w = img[0].size();
    int val = img[y][x];

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
                continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h)
            {
                if (img[ny][nx] == val)
                    same++;
            }
        }
    }

    return same;
}

inline double localAverage(const Image &img, int x, int y)
{
    int sum = 0, count = 0;
    int h = img.size(), w = img[0].size();

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h)
            {
                sum += img[ny][nx];
                count++;
            }
        }
    }

    return static_cast<double>(sum) / count;
}

inline int classifyPixel(int same)
{
    if (same == 8)
        return 0; // wnętrze
    else if (same >= 3)
        return 1; // brzeg
    else
        return 2; // narożnik
}

inline Vector2D extractFeatures(const Image &img, int x, int y)
{
    int same = countSameNeighbors(img, x, y);
    double avg = localAverage(img, x, y);
    double diff = img[y][x] - avg;
    return {static_cast<double>(same), diff};
}

inline void generateDatasetFromImage(const Image &img, std::vector<Vector2D> &samples, std::vector<int> &labels)
{
    // Na wszelki wypadek czyszczę te zmienne jeśli będę wywołać program w pętli.
    samples.clear();
    labels.clear();

    int h = img.size(), w = img[0].size();

    for (int y = 1; y < h - 1; ++y)
    {
        for (int x = 1; x < w - 1; ++x)
        {
            if (img[y][x] == 1)
            {
                Vector2D feat = extractFeatures(img, x, y);
                int same = countSameNeighbors(img, x, y);
                int label = classifyPixel(same);
                samples.push_back(feat);
                labels.push_back(label);
            }
        }
    }
}
