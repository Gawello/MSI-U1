#include <iostream>
#include <vector>
#include <fstream>
#include "../metoda/classifier.h"
#include "../problem/sample_data.h"

void trainClassifiers(std::vector<Classifier>& classifiers, const std::vector<Vector2D>& samples, const std::vector<int>& labels)
{
    for (int k = 0; k < 3; ++k)
    {
        classifiers[k].train(samples, labels, k);
    }
}

void testAndPrintResults(
    const std::vector<Classifier>& classifiers,
    const std::vector<Vector2D>& samples,
    const std::vector<int>& labels,
    std::ostream& out,
    bool printToConsole = true)
{
    int confusion[3][3] = {0};
    int correct = 0;

    for (size_t i = 0; i < samples.size(); ++i)
    {
        int expected = labels[i];
        int predicted = Classifier::predictClass(classifiers, samples[i]);

        confusion[expected][predicted]++;
        if (predicted == expected)
            correct++;

        out << "Sample " << i
            << " | expected: " << expected
            << " | predicted: " << predicted
            << (predicted == expected ? " ✅" : " ❌")
            << "\n";

        if (printToConsole)
        {
            std::cout << "Sample " << i
                      << " | expected: " << expected
                      << " | predicted: " << predicted
                      << (predicted == expected ? " ✅" : " ❌")
                      << std::endl;
        }
    }

    double accuracy = 100.0 * correct / samples.size();
    out << "\nAccuracy: " << accuracy << "%\n";

    out << "\nMacierz pomyłek:\n";
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            out << confusion[i][j] << " ";
        }
        out << "\n";
    }

    if (printToConsole)
    {
        std::cout << "\nAccuracy: " << accuracy << "%\n";
        std::cout << "\nMacierz pomyłek:\n";
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                std::cout << confusion[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
}

void saveWeightsToFile(std::ofstream& log, const std::vector<Classifier>& classifiers)
{
    log << "\nWektory wag dla każdej klasy:\n";
    for (size_t k = 0; k < classifiers.size(); ++k)
    {
        log << "Klasa " << k << ": ";
        const auto& w = classifiers[k].getWeights();
        for (double wi : w)
        {
            log << wi << " ";
        }
        log << "\n";
    }
}

int main()
{
    std::vector<Vector2D> samples;
    std::vector<int> labels;
    generateDatasetFromImage(testImage, samples, labels);

    std::vector<Classifier> classifiers(3);

    bool running = true;

    while (running)
    {
        std::cout << "\n=== MENU ===\n";
        std::cout << "1. Uruchom klasyfikację (na podstawie obrazu 2D)\n";
        std::cout << "2. Zapisz wyniki do pliku 'wyniki.txt'\n";
        std::cout << "0. Wyjście\n";
        std::cout << "Wybierz opcję: ";

        int option;
        std::cin >> option;

        switch (option)
        {
        case 1:
            trainClassifiers(classifiers, samples, labels);
            testAndPrintResults(classifiers, samples, labels, std::cout, true);
            break;

        case 2:
        {
            std::ofstream log("wyniki.txt");
            if (!log.is_open())
            {
                std::cerr << "Błąd przy otwieraniu pliku wynikowego!\n";
                break;
            }
            trainClassifiers(classifiers, samples, labels);
            saveWeightsToFile(log, classifiers);
            testAndPrintResults(classifiers, samples, labels, log, false);
            log.close();
            std::cout << "Wyniki zapisano do pliku 'wyniki.txt'\n";
            break;
        }

        case 0:
            running = false;
            break;

        default:
            std::cout << "Nieznana opcja. Spróbuj ponownie.\n";
            break;
        }
    }

    return 0;
}
