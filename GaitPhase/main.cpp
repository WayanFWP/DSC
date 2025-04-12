#include <iostream>
#include <vector>
#include "dataset.hpp"
#include "model.hpp"

Dataset dataset;
Dataset testCase;

int main() {
    if (!dataset.loadFromFile("data/dataset.txt")) return 1;

    MinMax normParams = dataset.normalize(); // normalize + simpan param
    std::vector<std::pair<std::vector<double>, int>> training_data;
    for (const auto& point : dataset.data)
        training_data.push_back({{point.x1, point.x2}, point.label});

    int data0 = 0, data1 = 0;
    for (const auto& d : dataset.data)
        (d.label == 0) ? data0++ : data1++;

    std::cout << "DataSet Label 0: " << data0 << ", Label 1: " << data1 << std::endl;

    MLP model;
    std::cout << "Training..." << std::endl;
    model.train(training_data, 5000, 0.1); // epochs bisa diubah
    std::cout << "Training selesai." << std::endl;

    if (!testCase.loadFromFile("test/testCase.txt")) return 1;

    testCase.applyNormalization(normParams); // normalize pakai data training
    std::vector<std::pair<std::vector<double>, int>> testData;
    for (const auto& point : testCase.data)
        testData.push_back({{point.x1, point.x2}, point.label});

    int correct = 0, test0 = 0, test1 = 0;
    for (const auto& t : testData)
        (t.second == 0) ? test0++ : test1++;
    std::cout << "TestCase Label 0: " << test0 << ", Label 1: " << test1 << std::endl;

    for (const auto& sample : testData) {
        double output = model.predict(sample.first[0], sample.first[1]);
        int predicted = (output > 0.5) ? 1 : 0;

        std::cout << "Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
                  << "\tTarget: " << sample.second
                  << "\tPredicted: " << predicted
                  << "\tOutput: " << output << std::endl;

        if (predicted == sample.second) correct++;
    }

    double accuracy = 100.0 * correct / testData.size();
    std::cout << "\nAkurasi: " << accuracy << "%" << std::endl;

    return 0;
}
