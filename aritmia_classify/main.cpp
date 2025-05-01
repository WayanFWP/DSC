#include <iostream>
#include <vector>

#include "dataset.hpp"
#include "model.hpp"

Dataset dataset;
Dataset testCase;

void testClassifyArrhythmia() {
  std::cout << "Testing classifyArrhythmia function:" << std::endl;

  // Test data points (RR, QRSd, expected label)
  std::vector<std::tuple<double, double, int>> test_data = {
      {0.8, 90, 0},    // Normal
      {0.2, 90, 3},    // Tachycardia
      {0.4, 90, 2},    // R-on-T
      {0.4, 110, 6},   // PVC
      {0.8, 110, 5},   // Fusion
      {1.4, 90, 4},    // Bradycardia
      {1.8, 90, 1}     // Dropped
  };

  for (const auto& data_point : test_data) {
    double rr   = std::get<0>(data_point);
    double qrsd = std::get<1>(data_point);
    int    expected_label = std::get<2>(data_point);
    int    actual_label   = dataset.classifyArrhythmia(rr, qrsd);

    std::cout << "RR: " << rr << ", QRSd: " << qrsd << ", Expected: " << expected_label
              << ", Actual: " << actual_label << std::endl;
  }
}

int main() {
  std::srand(std::time(0));
  if (!dataset.loadFromFile("data/dataset.txt"))
    return 1;

  testClassifyArrhythmia(); // Call the testing function

  std::vector<std::pair<std::vector<double>, int>> training_data;
  for (const auto& point : dataset.data) training_data.push_back({{point.RR, point.QRSd}, point.label});

  int class_counts[7] = {0};
  for (const auto& d : dataset.data) {
      if (d.label >= 0 && d.label < 7) class_counts[d.label]++;
  }

  for (int i = 0; i < 7; ++i) {
      std::cout << "Label " << i << ": " << class_counts[i] << " data\n";
  }
  MLP model;
  std::cout << "Training..." << std::endl;
  model.train(training_data, 200, 0.1);  // epoch 500, learning rate 0.1
  std::cout << "Training section done." << std::endl;

  if (!testCase.loadFromFile("test/testCase.txt"))
    return 1;

  testCase.addNoiseToData(0.7);  // Add noise to test data

  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto& point : testCase.data) testData.push_back({{point.RR, point.QRSd}, point.label});

  int correct = 0, test0 = 0, test1 = 0;
  for (const auto& t : testData) (t.second == 0) ? test0++ : test1++;
  std::cout << "TestCase Label 0: " << test0 << ", Label 1: " << test1 << std::endl;

  int count = 1;

  for (const auto& sample : testData) {
    std::vector<double> output_vector = model.predict(sample.first[0], sample.first[1]);
  
    // Find the index of the maximum value in the output vector
    int predicted = std::distance(output_vector.begin(), std::max_element(output_vector.begin(), output_vector.end()));
  
    std::string predictLabel;
    switch (predicted) {
      case 0: predictLabel = "Normal"; break;
      case 1: predictLabel = "Dropped"; break;
      case 2: predictLabel = "R-on-T"; break;
      case 3: predictLabel = "Tachycardia"; break;
      case 4: predictLabel = "Bradycardia"; break;
      case 5: predictLabel = "Fusion"; break;
      case 6: predictLabel = "PVC"; break;
      default: predictLabel = "Unknown"; break;
    }
  
    std::string targetLabel;
    switch (sample.second) {
      case 0: targetLabel = "Normal"; break;
      case 1: targetLabel = "Dropped"; break;
      case 2: targetLabel = "R-on-T"; break;
      case 3: targetLabel = "Tachycardia"; break;
      case 4: targetLabel = "Bradycardia"; break;
      case 5: targetLabel = "Fusion"; break;
      case 6: targetLabel = "PVC"; break;
      default: targetLabel = "Unknown"; break;
    }
  
    std::cout << count << " Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
              << "\n|>=============================> Target: " << targetLabel << "\tPredicted: " << predictLabel
              << "\tOutput: " << output_vector[predicted] << std::endl; // Show probability of predicted class
  
    if (predicted == sample.second) correct++;
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