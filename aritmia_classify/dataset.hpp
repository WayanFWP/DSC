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
  double RR;
  double QRSd;
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
      double rr, qrsd;
      if (iss >> rr >> qrsd) {
        int label = classifyArrhythmia(rr, qrsd);
        data.push_back({rr, qrsd, label});
      }
    }

    file.close();
    return true;
  }

  int classifyArrhythmia(double rr, double qrsd) {
    // Normal: RR 0.6-1.2 ; QRSd 80-100
    if (rr >= 0.6 && rr <= 1.2 && qrsd >= 80 && qrsd <= 100) {
      return 0; // Normal
    }
    // Region 1: RR < 0.32 (Tachycardia)
    else if (rr < 0.32) {
      return 3; // Tachycardia
    }
    // Region 2: normal QRSd; 0.32 < RR < 0.60 (R-on-T)
    else if (rr >= 0.32 && rr < 0.60 && qrsd >= 80 && qrsd <= 100) {
      return 2; // R-on-T
    }
    // Region 3: QRSd > 100; 0.32 < RR < 0.60 (PVC - needs context)
    else if (rr >= 0.32 && rr < 0.60 && qrsd > 100) {
      return 6; // PVC (context check needed later)
    }
    // Region 4: Normal RR; QRSd > 100 (Fusion)
    else if (rr >= 0.6 && rr <= 1.2 && qrsd > 100) {
      return 5; // Fusion
    }
    // Region 5: 1.2 < RR < 1.66 (Bradycardia - needs consecutive points)
    else if (rr > 1.2 && rr <= 1.66) {
      return 4; // Bradycardia candidate (context check needed later)
    }
    // Region 6: RR > 1.66 (Dropped)
    else if (rr > 1.66) {
      return 1; // Dropped
    }
  }

  void addNoiseToData(double noise_factor = 0.01) {
    for (auto& d : data) {
      d.RR += (rand() % 2001 - 1000) / 100000.0 * noise_factor;
      d.QRSd += (rand() % 2001 - 1000) / 100000.0 * noise_factor;
    }
  }

  void duplicateAndAugment(int label, int times = 2) {
    std::vector<DataPoint> augmented;
    for (const auto& d : data) {
      if (d.label == label) { // Only duplicate data points with the specified label
        for (int i = 0; i < times; ++i) {
          DataPoint aug = d;
          aug.RR += ((rand() % 2001 - 1000) / 1000.0) * 0.01;
          aug.QRSd += ((rand() % 2001 - 1000) / 1000.0) * 0.01;
          augmented.push_back(aug);
        }
      }
    }
    data.insert(data.end(), augmented.begin(), augmented.end());
  }
};
#endif
