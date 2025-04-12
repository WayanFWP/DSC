#include <iostream>
#include <vector>

#include "dataset.hpp"
#include "model.hpp"

Dataset dataset;
Dataset testCase;

int main() {
  std::srand(std::time(0));  // initiate random seed

  if (!dataset.loadFromFile("data/labeled_dataset.txt"))
    return 1;

  std::vector<std::pair<std::vector<double>, int>> training_data;
  for (const auto& point : dataset.data) training_data.push_back({{point.x1, point.x2}, point.label});

  int data0 = 0, data1 = 0;
  for (const auto& d : dataset.data) (d.label == 0) ? data0++ : data1++;

  std::cout << "DataSet Label 0: " << data0 << ", Label 1: " << data1 << std::endl;

  MLP model;
  std::cout << "Training..." << std::endl;
  model.train(training_data, 10000, 0.1);  // epoch 10000, learning rate 0.1
  std::cout << "Training selesai." << std::endl;

  if (!testCase.loadFromFile("test/testCase.txt"))
    return 1;

  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto& point : testCase.data) testData.push_back({{point.x1, point.x2}, point.label});

  int correct = 0, test0 = 0, test1 = 0;
  for (const auto& t : testData) (t.second == 0) ? test0++ : test1++;
  std::cout << "TestCase Label 0: " << test0 << ", Label 1: " << test1 << std::endl;

  while (true) {
    int correct = 0;
    int count   = 1;

    for (const auto& sample : testData) {
      double output    = model.predict(sample.first[0], sample.first[1]);
      int    predicted = (output > 0.5) ? 1 : 0;

      std::string predictLabel = (predicted == 0) ? "stance (0)" : "swing (1)";
      std::string targetLabel  = (sample.second == 0) ? "stance (0)" : "swing (1)";

      std::cout << count << " Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
                << "\tTarget: " << targetLabel << "\tPredicted: " << predictLabel << "\tOutput: " << output << std::endl;

      if (predicted == sample.second)
        correct++;
      count++;
    }
    double accuracy = 100.0 * correct / testData.size();

    std::cout << "\nDo you want to see the accuracy? (a): ";
    char choice;
    std::cin >> choice;
    if (choice == 'q' || choice == 'n')
      break;
    if (choice == 'a')
      std::cout << "\nAccuracy: " << accuracy << "%" << std::endl;
  }

  return 0;
}
