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

  int class_counts[6] = {0};
  for (const auto& d : dataset.data) {
      if (d.label >= 0 && d.label < 6) class_counts[d.label]++;
  }

  for (int i = 0; i < 6; ++i) {
      std::cout << "Label " << i << ": " << class_counts[i] << " data\n";
  }
  
  MLP model;
  std::cout << "Training..." << std::endl;
  model.train(training_data, 200, 0.1);  // epoch 500, learning rate 0.1
  std::cout << "Training section done." << std::endl;

  if (!testCase.loadFromFile("test/recall.txt"))
    return 1;

  testCase.addNoiseToData(0.7);  // Add noise to test data

  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto& point : testCase.data) testData.push_back({{point.x1, point.x2}, point.label});

  int class_counter[6] = {0};
  for (const auto& d : testCase.data) {
      if (d.label >= 0 && d.label < 6) class_counter[d.label]++;
  }

  for (int i = 0; i < 6; ++i) {
      std::cout << "Label " << i << ": " << class_counter[i] << " data\n";
  }

  int correct = 0;
  int count = 1;

  for (const auto& sample : testData) {
    std::vector<double> output_vector = model.predict(sample.first[0], sample.first[1]);
    int predicted = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));

    std::string predictLabel;
    switch (predicted) {
      case 0: predictLabel = "IC"; break;
      case 1: predictLabel = "FF"; break;
      case 2: predictLabel = "HO"; break;
      case 3: predictLabel = "MSt"; break;
      case 4: predictLabel = "TO"; break;
      case 5: predictLabel = "Sw"; break;
      default: predictLabel = "Unknown"; break;
    }

    std::string targetLabel;
    switch (sample.second) {
      case 0: targetLabel = "IC"; break;
      case 1: targetLabel = "FF"; break;
      case 2: targetLabel = "HO"; break;
      case 3: targetLabel = "MSt"; break;
      case 4: targetLabel = "TO"; break;
      case 5: targetLabel = "Sw"; break;
      default: targetLabel = "Unknown"; break;
    }

    std::cout << count << " Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
              << "\n|>=============================> Target: " << targetLabel << "\tPredicted: " << predictLabel << "\tOutput: ";
    for (size_t i = 0; i < output_vector.size(); ++i) {
        std::string label;
        switch (i) {
          case 0: label = "IC"; break;
          case 1: label = "FF"; break;
          case 2: label = "HO"; break;
          case 3: label = "MSt"; break;
          case 4: label = "TO"; break;
          case 5: label = "Sw"; break;
          default: label = "Unknown"; break;
        }
        std::cout << label << " (" << output_vector[i] * 100 << "%) ";
    }
    std::cout << std::endl;

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
