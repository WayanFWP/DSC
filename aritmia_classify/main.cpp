#include <iostream>
#include <vector>

#include "dataset.hpp"
#include "model.hpp"

Dataset dataset;
Dataset testCase;

int main() {
  std::srand(std::time(0));
  if (!dataset.loadFromFile("data/dataset.txt"))
    return 1;

  std::vector<std::pair<std::vector<double>, int>> training_data;
  for (const auto& point : dataset.data) training_data.push_back({{point.x1, point.x2}, point.label});

  int data0 = 0, data1 = 0;
  for (const auto& d : dataset.data) (d.label == 0) ? data0++ : data1++;

  std::cout << "DataSet Label 0: " << data0 << ", Label 1: " << data1 << std::endl;

  MLP model;
  std::cout << "Training..." << std::endl;
  model.train(training_data, 200, 0.1);  // epoch 500, learning rate 0.1
  std::cout << "Training section done." << std::endl;

  if (!testCase.loadFromFile("test/testCase.txt"))
    return 1;

  testCase.addNoiseToData(0.7);  // Add noise to test data

  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto& point : testCase.data) testData.push_back({{point.x1, point.x2}, point.label});

  int correct = 0, test0 = 0, test1 = 0;
  for (const auto& t : testData) (t.second == 0) ? test0++ : test1++;
  std::cout << "TestCase Label 0: " << test0 << ", Label 1: " << test1 << std::endl;

  int count = 1;

  for (const auto& sample : testData) {
    double output    = model.predict(sample.first[0], sample.first[1]);
    int    predicted = (output > 0.5) ? 1 : 0;

    std::string predictLabel = (predicted == 0) ? "stance (0)" : "swing (1)";
    std::string targetLabel  = (sample.second == 0) ? "stance (0)" : "swing (1)";

    std::cout << count << " Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
              << "\n|>=============================> Target: " << targetLabel << "\tPredicted: " << predictLabel << "\tOutput: " << output << std::endl;

    if (predicted == sample.second)
      correct++;
    count++;
  }
  double accuracy = 100.0 * correct / testData.size();

  while (true) {
    std::cout << "\nDo you want to see the accuracy? (a): ";
    char choice;
    std::cin >> choice;
    if (choice == 'a')
      std::cout << "\nAccuracy: " << accuracy << "%" << std::endl;
    else
      break;
  }

  return 0;
}
