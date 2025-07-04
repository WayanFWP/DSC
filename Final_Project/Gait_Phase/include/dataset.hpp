#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct DataPoint {
  double x1;
  double x2;
  int    label;
};

class Dataset {
 public:
  std::vector<DataPoint> data;

  // Baca file data (x1 x2 label)
  bool loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << path << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      double             x1, x2;
      if (iss >> x1 >> x2) {
        int label = classifyPhase(x1, x2);  // ✅ Auto-label using classifyPhase
        data.push_back({x1, x2, label});
      }
    }

    file.close();
    return true;
  }

  // Add to your Dataset class definition
  void addDataPoint(double x1, double x2, int label) { data.push_back({x1, x2, label}); }

  void balanceDataset(Dataset& dataset) {
  int class_counts[6] = {0};
  for (const auto& point : dataset.data) {
    if (point.label >= 0 && point.label < 6) {
      class_counts[point.label]++;
    }
  }

  int max_count = *std::max_element(class_counts, class_counts + 6);

  for (int label = 0; label < 6; ++label) {
    int augment_factor = (max_count - class_counts[label]) / class_counts[label];
    if (augment_factor > 0) {
      for (const auto& point : dataset.data) {
        if (point.label == label) {
          for (int i = 0; i < augment_factor; ++i) {
            dataset.addDataPoint(point.x1, point.x2, point.label);
          }
        }
      }
    }
  }
}

  void dataToClassify() {
    double x1, x2;
    for (const auto& point : data) {
      x1 = point.x1;
      x2 = point.x2;

      int label = classifyPhase(x1, x2);  // ✅ Auto-label using classifyPhase
      data.push_back({x1, x2, label});
    }
  }

  void addNoiseToData(double noise_factor = 0.01) {
    for (auto& d : data) {
      d.x1 += (rand() % 2001 - 1000) / 100000.0 * noise_factor;
      d.x2 += (rand() % 2001 - 1000) / 100000.0 * noise_factor;
    }
  }

  void duplicateAndAugment(int times = 2) {
    std::vector<DataPoint> augmented;
    for (const auto& d : data) {
      for (int i = 0; i < times; ++i) {
        DataPoint aug = d;
        aug.x1 += ((rand() % 2001 - 1000) / 100000.0) * 0.5;
        aug.x2 += ((rand() % 2001 - 1000) / 100000.0) * 0.5;
        augmented.push_back(aug);
      }
    }
    data.insert(data.end(), augmented.begin(), augmented.end());
  }

  int classifyPhase(double heel, double toe) {
    if (heel < 0.03 && toe < 0.03)
      return 5;  // swing
    else if (heel > 1.5 && toe >= 0.3 && toe <= 0.5)
      return 1;  // foot flat
    else if (heel > 1.0 && toe < 0.05)
      return 0;  // initial contact
    else if (heel < 0.05 && toe > 0.44)
      return 2;  // heel off
    else if (heel < 0.03 && toe >= 0.09 && toe <= 0.6)
      return 4;  // toe off
    else if (heel >= 0.5 && toe > 0.2)
      return 3;  // mid stance
    else
      return 5;  // default to swing
  }
};

#endif
