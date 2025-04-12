#include <iostream>
#include <vector>

#include "dataset.hpp"
#include "model.hpp"

Dataset dataset;   // Data untuk pengujian
Dataset testCase;  // Data untuk pelatihan

int main() {
  std::srand(std::time(0));  // Inisialisasi seed untuk random number generator

  if (!dataset.loadFromFile("data/labeled_dataset.txt"))
    return 1;

  // Normalisasi data pelatihan dan simpan parameter min/max
  // dataset.balanceWithSMOTE();
  dataset.normalize();
  MinMax normParams = dataset.normalize();  // Normalisasi + simpan parameter min/max
  
  std::vector<std::pair<std::vector<double>, int>> training_data;
  for (const auto& point : dataset.data) training_data.push_back({{point.x1, point.x2}, point.label});
  
  int data0 = 0, data1 = 0;
  for (const auto& d : dataset.data) (d.label == 0) ? data0++ : data1++;

  std::cout << "DataSet Label 0: " << data0 << ", Label 1: " << data1 << std::endl;
  
  MLP model(0.1);
  std::cout << "Training..." << std::endl;
  model.train(training_data, 10000, 0.1);  // epoch 3000, learning rate 0.05
  std::cout << "Training selesai." << std::endl;
  
  if (!testCase.loadFromFile("test/testCase.txt"))
  return 1;
  
  testCase.applyNormalization(normParams);  // Terapkan normalisasi yang sama ke test set
  testCase.addNoiseToData(0.05);

  std::vector<std::pair<std::vector<double>, int>> testData;
  for (const auto& point : testCase.data) testData.push_back({{point.x1, point.x2}, point.label});

  int correct = 0, test0 = 0, test1 = 0;
  for (const auto& t : testData) (t.second == 0) ? test0++ : test1++;
  std::cout << "TestCase Label 0: " << test0 << ", Label 1: " << test1 << std::endl;

  int count = 0;

  for (const auto& sample : training_data) {
    double output    = model.predict(sample.first[0], sample.first[1]);
    int    predicted = (output > 0.5) ? 1 : 0;

    std::string predictLabeles  = (predicted == 0) ? "stance (0)" : "swing (1)";
    std::string targetedLabeles = (sample.second == 0) ? "stance (0)" : "swing (1)";

    std::cout << count << " Input: (" << sample.first[0] << ", " << sample.first[1] << ")  "
              << "\tTarget: " << targetedLabeles << "\tPredicted: " << predictLabeles << "\tOutput: " << output << std::endl;

    if (predicted == sample.second)
      correct++;
    count++;
  }

  double accuracy = 100.0 * correct / testData.size();
  std::cout << "\nAkurasi: " << accuracy << "%" << std::endl;

  return 0;
}
