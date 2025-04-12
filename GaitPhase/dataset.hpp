#ifndef DATASET_HPP
#define DATASET_HPP

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

struct MinMax {
  double minX1, maxX1;
  double minX2, maxX2;
};

class Dataset {
 public:
  std::vector<DataPoint> data;

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
      int                label;
      if (iss >> x1 >> x2 >> label) {
        data.push_back({x1, x2, label});
      }
    }

    file.close();
    return true;
  }

  MinMax normalize() {
    MinMax mm;
    if (data.empty())
      return mm;

    mm.minX1 = mm.maxX1 = data[0].x1;
    mm.minX2 = mm.maxX2 = data[0].x2;

    for (const auto& d : data) {
      if (d.x1 < mm.minX1)
        mm.minX1 = d.x1;
      if (d.x1 > mm.maxX1)
        mm.maxX1 = d.x1;
      if (d.x2 < mm.minX2)
        mm.minX2 = d.x2;
      if (d.x2 > mm.maxX2)
        mm.maxX2 = d.x2;
    }

    for (auto& d : data) {
      d.x1 = (d.x1 - mm.minX1) / (mm.maxX1 - mm.minX1);
      d.x2 = (d.x2 - mm.minX2) / (mm.maxX2 - mm.minX2);
    }

    return mm;
  }

  void applyNormalization(const MinMax& mm) {
    for (auto& d : data) {
      d.x1 = (d.x1 - mm.minX1) / (mm.maxX1 - mm.minX1);
      d.x2 = (d.x2 - mm.minX2) / (mm.maxX2 - mm.minX2);
    }
  }
};

#endif
