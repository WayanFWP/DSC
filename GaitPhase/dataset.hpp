#ifndef DATASET_HPP
#define DATASET_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

struct DataPoint {
    double x1;
    double x2;
    int label;
};

struct MinMax {
    double minX1, maxX1;
    double minX2, maxX2;
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
            double x1, x2;
            int label;
            if (iss >> x1 >> x2 >> label) {
                data.push_back({x1, x2, label});
            }
        }

        file.close();
        return true;
    }

    // Normalisasi data ke [0,1] dan simpan nilai min/max
    MinMax normalize() {
        MinMax mm;
        if (data.empty()) return mm;

        mm.minX1 = mm.maxX1 = data[0].x1;
        mm.minX2 = mm.maxX2 = data[0].x2;

        for (const auto& d : data) {
            if (d.x1 < mm.minX1) mm.minX1 = d.x1;
            if (d.x1 > mm.maxX1) mm.maxX1 = d.x1;
            if (d.x2 < mm.minX2) mm.minX2 = d.x2;
            if (d.x2 > mm.maxX2) mm.maxX2 = d.x2;
        }

        for (auto& d : data) {
            d.x1 = (d.x1 - mm.minX1) / (mm.maxX1 - mm.minX1);
            d.x2 = (d.x2 - mm.minX2) / (mm.maxX2 - mm.minX2);
        }

        return mm;
    }

    // Terapkan normalisasi berdasarkan parameter yang sudah ada
    void applyNormalization(const MinMax& mm) {
        for (auto& d : data) {
            d.x1 = std::max(0.0, std::min(1.0, (d.x1 - mm.minX1) / (mm.maxX1 - mm.minX1)));
            d.x2 = std::max(0.0, std::min(1.0, (d.x2 - mm.minX2) / (mm.maxX2 - mm.minX2)));
        }
    }

    // SMOTE sederhana untuk menyeimbangkan minoritas
    void balanceWithSMOTE(int targetPerClass) {
        std::vector<DataPoint> label0, label1;
        for (const auto& d : data) {
            if (d.label == 0) label0.push_back(d);
            else label1.push_back(d);
        }

        int minorLabel = (label0.size() < label1.size()) ? 0 : 1;
        std::vector<DataPoint>& minor = (minorLabel == 0) ? label0 : label1;
        int n_need = targetPerClass - minor.size();
        if (n_need <= 0) return;

        std::srand(std::time(0));
        std::vector<DataPoint> synthetic;
        for (int i = 0; i < n_need; ++i) {
            const DataPoint& a = minor[rand() % minor.size()];
            const DataPoint& b = minor[rand() % minor.size()];

            double t = (double)rand() / RAND_MAX;
            double new_x1 = a.x1 + t * (b.x1 - a.x1);
            double new_x2 = a.x2 + t * (b.x2 - a.x2);

            synthetic.push_back({new_x1, new_x2, minorLabel});
        }

        data.insert(data.end(), synthetic.begin(), synthetic.end());
    }

    // Menambahkan noise acak kecil pada data
    void addNoiseToData(double noise_factor = 0.01) {
        for (auto& d : data) {
            // Tambahkan noise acak kecil ke setiap fitur
            d.x1 += (rand() % 2001 - 1000) / 100000.0 * noise_factor; // Random noise antara -0.01 dan 0.01
            d.x2 += (rand() % 2001 - 1000) / 100000.0 * noise_factor; // Random noise antara -0.01 dan 0.01
        }
    }
};

#endif
