#ifndef DATASET_HPP
#define DATASET_HPP

#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Function to load a 12x12 matrix from a file
vector<vector<double>> load_matrix(const string& filename) {
    vector<vector<double>> matrix(12, vector<double>(12, 0));
    ifstream file("dataset/" + filename);
    if (!file) {
        cerr << "Error: Cannot open file dataset/" << filename << endl;
        exit(1);
    }

    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            file >> matrix[i][j];

    file.close();
    return matrix;
}

// Function to generate a noisy variation of a letter
vector<vector<double>> generate_variation(const vector<vector<double>>& base, double noise_prob) {
    vector<vector<double>> variant = base;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            if (dist(gen) < noise_prob) {
                variant[i][j] = (variant[i][j] == 1) ? 0 : 1;  // Flip some pixels
            }
        }
    }
    return variant;
}

// Function to load dataset dynamically
void load_dataset(vector<vector<vector<double>>>& letters, vector<vector<vector<double>>>& variations) {
    vector<string> filenames = {"A.txt", "B.txt", "C.txt", "D.txt", "E.txt", "F.txt", "G.txt", "H.txt"};

    for (const string& file : filenames) {
        vector<vector<double>> letter = load_matrix(file);
        letters.push_back(letter);
        variations.push_back(generate_variation(letter, 0.05)); // 5% noise
        variations.push_back(generate_variation(letter, 0.10)); // 10% noise
    }
}

// Function to flatten a 12x12 matrix into a 1D vector
inline vector<double> flatten(const vector<vector<double>>& letter) {
    vector<double> flat;
    for (const auto& row : letter) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Function to get the target output (one-hot encoding)
inline vector<double> get_target(char letter) {
    vector<double> target(8, 0);
    target[letter - 'A'] = 1;
    return target;
}

// Function to load a 12x12 matrix from test case file
vector<vector<double>> load_test_matrix(const string& filename) {
    vector<vector<double>> matrix(12, vector<double>(12, 0));
    ifstream file("testcase/" + filename); // Load file from test directory
    if (!file) {
        cerr << "Error: Cannot open file testcase/" << filename << endl;
        exit(1);
    }

    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 12; ++j)
            file >> matrix[i][j];

    file.close();
    return matrix;
}

#endif // DATASET_HPP
