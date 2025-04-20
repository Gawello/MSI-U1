#include <iostream>
#include <vector>
#include <fstream>
#include "../metoda/classifier.h"
#include "../problem/sample_data.h"



int main(int argc, char const *argv[])
{
    std::vector<Vector2D> samples;
    std::vector<int> labels;
    generateDatasetFromImage(testImage, samples, labels);

    std::ofstream log("wyniki.txt");
    if (!log.is_open())
    {
        std::cerr << "Błąd przy otwieraniu pliku wynikowego!" << std::endl;
        return 1;
    }

    std::vector<Classifier> classifiers(3);
    // Macierz pomyłek
    int confusion[3][3] = {0}; // confusion[expected][predicted]

    // Trening dla każdej klasy (1-vs-rest)
    for (int k = 0; k < 3; ++k)
    {
        classifiers[k].train(samples, labels, k);
    }

    // Zapis wektora wag do pliku
    log << "\nWektory wag dla każdej klasy:\n";

    for (size_t k = 0; k < classifiers.size(); ++k)
    {
        log << "Klasa " << k << ": ";
        const auto &w = classifiers[k].getWeights();
        for (double wi : w)
        {
            log << wi << " ";
        }
        log << "\n";
    }

    // Testowanie i ocena
    int correct = 0;
    for (int i = 0; i < samples.size(); ++i)
    {
        int expected = labels[i];
        int predicted = Classifier::predictClass(classifiers, samples[i]);
        confusion[expected][predicted]++;

        // Drukowanie wyników w terminalu
        std::cout << "Sample " << i
                  << " | expected: " << expected
                  << " | predicted: " << predicted
                  << (predicted == expected ? " ✅" : " ❌")
                  << std::endl;

        // Zapis wyników do pliku
        log << "Sample " << i
            << " | expected: " << expected
            << " | predicted: " << predicted
            << (predicted == expected ? " ✅" : " ❌")
            << "\n";

        if (predicted == expected)
            correct++;
    }

    double accuracy = 100.0 * correct / samples.size();
    std::cout << "\nAccuracy: " << accuracy << "%" << std::endl;

    // Drukowanie macierzy pomyłek w terminalu
    std::cout << "\nMacierz pomyłek:\n";
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout << confusion[i][j] << " ";
        }
        std::cout << "\n";
    }

    // Zapis macierzy pomyłek do pliku
    log << "\nMacierz pomyłek:\n";
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            log << confusion[i][j] << " ";
        }
        log << "\n";
    }

    return 0;
}
