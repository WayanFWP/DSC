#ifndef DATASET_HPP
#define DATASET_HPP

#include <algorithm>
#include <cstdlib>
#include <ctime>
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
      std::cerr << "Gagal membuka file: " << path << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      double             x1, x2;
      if (iss >> x1 >> x2) {
        int label = (x1 < 0.1 && x2 < 0.1) ? 0 : 1;  // ✅ Auto-label
        data.push_back({x1, x2, label});
      }
    }

    file.close();
    return true;
  }

  // Menambahkan noise acak kecil pada data
  void addNoiseToData(double noise_factor = 0.01) {
    for (auto& d : data) {
      // Tambahkan noise acak kecil ke setiap fitur
      d.x1 += (rand() % 2001 - 1000) / 100000.0 * noise_factor;  // Random noise antara -0.01 dan 0.01
      d.x2 += (rand() % 2001 - 1000) / 100000.0 * noise_factor;  // Random noise antara -0.01 dan 0.01
    }
  }

  void duplicateAndAugment(int times = 2) {
    std::vector<DataPoint> augmented;
    for (const auto& d : data) {
      for (int i = 0; i < times; ++i) {
        DataPoint aug = d;
        aug.x1 += ((rand() % 2001 - 1000) / 1000.0) * 0.01;
        aug.x2 += ((rand() % 2001 - 1000) / 1000.0) * 0.01;
        augmented.push_back(aug);
      }
    }
    data.insert(data.end(), augmented.begin(), augmented.end());
  }
};

#endif
