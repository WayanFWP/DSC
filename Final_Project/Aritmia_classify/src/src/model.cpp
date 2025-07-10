#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>   // Add this for std::setprecision
#include <iostream>  // Add this for std::cout
#include <numeric>
#include <random>

// Model::Model(int inputSize, int hiddenSize, int outputSize, double lr)
//     : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(lr)
// {
//     std::srand(std::time(0));

//     W1 = std::vector<std::vector<double>>(hiddenSize, std::vector<double>(inputSize));
//     b1 = std::vector<double>(hiddenSize);
//     W2 = std::vector<std::vector<double>>(outputSize, std::vector<double>(hiddenSize));
//     b2 = std::vector<double>(outputSize);

//     // Random init
//     for (auto &row : W1)
//         for (auto &val : row)
//             val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

//     for (auto &row : W2)
//         for (auto &val : row)
//             val = (double(std::rand()) / RAND_MAX - 0.5) * 2;
// }

Model::Model(int inputSize, int hiddenSize1, int hiddenSize2, int hiddenSize3, int hiddenSize4, int outputSize, double lr)
    : inputSize(inputSize),
      hiddenSize1(hiddenSize1),
      hiddenSize2(hiddenSize2),
      hiddenSize3(hiddenSize3),
      hiddenSize4(hiddenSize4),
      outputSize(outputSize),
      learningRate(lr) {
  std::srand(std::time(0));

  // Initialize weight matrices
  W1 = std::vector<std::vector<double>>(hiddenSize1, std::vector<double>(inputSize));
  W2 = std::vector<std::vector<double>>(hiddenSize2, std::vector<double>(hiddenSize1));
  W3 = std::vector<std::vector<double>>(hiddenSize3, std::vector<double>(hiddenSize2));
  W4 = std::vector<std::vector<double>>(hiddenSize4, std::vector<double>(hiddenSize3));
  W5 = std::vector<std::vector<double>>(outputSize, std::vector<double>(hiddenSize4));

  // Initialize bias vectors
  b1 = std::vector<double>(hiddenSize1);
  b2 = std::vector<double>(hiddenSize2);
  b3 = std::vector<double>(hiddenSize3);
  b4 = std::vector<double>(hiddenSize4);
  b5 = std::vector<double>(outputSize);

  // Random initialization for all weights
  for (auto &row : W1)
    for (auto &val : row) val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

  for (auto &row : W2)
    for (auto &val : row) val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

  for (auto &row : W3)
    for (auto &val : row) val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

  for (auto &row : W4)
    for (auto &val : row) val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

  for (auto &row : W5)
    for (auto &val : row) val = (double(std::rand()) / RAND_MAX - 0.5) * 2;
}

double Model::sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

double Model::sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

std::vector<double> Model::softmax(const std::vector<double> &x) {
  double              maxVal = *std::max_element(x.begin(), x.end());
  std::vector<double> exps(x.size());
  double              sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    exps[i] = std::exp(x[i] - maxVal);
    sum += exps[i];
  }
  for (double &val : exps) val /= sum;
  return exps;
}

double Model::dot(const std::vector<double> &a, const std::vector<double> &b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
  return result;
}

void Model::add_to(std::vector<double> &a, const std::vector<double> &b, double scale) {
  for (size_t i = 0; i < a.size(); ++i) a[i] += b[i] * scale;
}

void Model::add_to(std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, double scale) {
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < a[i].size(); ++j) a[i][j] += b[i][j] * scale;
}

// void Model::train(const std::vector<std::vector<double>> &X,
//                   const std::vector<std::vector<double>> &Y,
//                   int epochs)
// {
//     size_t n = X.size();
//     for (int epoch = 0; epoch < epochs; ++epoch)
//     {
//         double totalMSE = 0;

//         for (size_t i = 0; i < n; ++i)
//         {
//             const auto &x = X[i];
//             const auto &y = Y[i];

//             // ---- Forward ----
//             std::vector<double> z1(hiddenSize), a1(hiddenSize);
//             for (int h = 0; h < hiddenSize; ++h)
//             {
//                 z1[h] = dot(W1[h], x) + b1[h];
//                 a1[h] = sigmoid(z1[h]);
//             }

//             std::vector<double> z2(outputSize), y_hat(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//                 z2[o] = dot(W2[o], a1) + b2[o];
//             y_hat = softmax(z2);

//             // ---- Loss ----
//             std::vector<double> error(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//             {
//                 error[o] = y_hat[o] - y[o];
//                 totalMSE += 0.5 * error[o] * error[o];
//             }

//             // ---- Backprop ----
//             std::vector<std::vector<double>> dW2(outputSize, std::vector<double>(hiddenSize));
//             std::vector<double> db2(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//             {
//                 db2[o] = error[o];
//                 for (int h = 0; h < hiddenSize; ++h)
//                     dW2[o][h] = error[o] * a1[h];
//             }

//             std::vector<double> dA1(hiddenSize, 0.0);
//             for (int h = 0; h < hiddenSize; ++h)
//                 for (int o = 0; o < outputSize; ++o)
//                     dA1[h] += W2[o][h] * error[o];

//             std::vector<std::vector<double>> dW1(hiddenSize, std::vector<double>(inputSize));
//             std::vector<double> db1(hiddenSize);
//             for (int h = 0; h < hiddenSize; ++h)
//             {
//                 double dZ1 = dA1[h] * sigmoid_derivative(z1[h]);
//                 db1[h] = dZ1;
//                 for (int j = 0; j < inputSize; ++j)
//                     dW1[h][j] = dZ1 * x[j];
//             }

//             // ---- Update ----
//             add_to(W2, dW2, -learningRate);
//             add_to(b2, db2, -learningRate);
//             add_to(W1, dW1, -learningRate);
//             add_to(b1, db1, -learningRate);
//         }
//         int correct = 0;
//         for (size_t i = 0; i < n; ++i)
//         {
//             int pred = predict(X[i]);
//             int actual = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
//             if (pred == actual)
//                 correct++;
//         }
//         double acc = 100.0 * correct / n;

//         mseHistory.push_back(totalMSE / n);
//         accuracyHistory.push_back(acc);
//     }
// }

// void Model::train(const std::vector<std::vector<double>> &X,
//                   const std::vector<std::vector<double>> &Y,
//                   int epochs)
// {
//     size_t n = X.size();
//     int epoch = 0;
//     double avgMSE = 1.0; // Initialize with a value > 1e-6

//     // while (avgMSE > 1e-6 && epoch < epochs) // Add epoch limit as safety
//     while (avgMSE > 1e-6) // Add epoch limit as safety
//     {
//         double totalMSE = 0;

//         for (size_t i = 0; i < n; ++i)
//         {
//             const auto &x = X[i];
//             const auto &y = Y[i];

//             // ---- Forward ----
//             std::vector<double> z1(hiddenSize), a1(hiddenSize);
//             for (int h = 0; h < hiddenSize; ++h)
//             {
//                 z1[h] = dot(W1[h], x) + b1[h];
//                 a1[h] = sigmoid(z1[h]);
//             }

//             std::vector<double> z2(outputSize), y_hat(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//                 z2[o] = dot(W2[o], a1) + b2[o];
//             y_hat = softmax(z2);

//             // ---- Loss ----
//             std::vector<double> error(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//             {
//                 error[o] = y_hat[o] - y[o];
//                 totalMSE += 0.5 * error[o] * error[o];
//             }

//             // ---- Backprop ----
//             std::vector<std::vector<double>> dW2(outputSize, std::vector<double>(hiddenSize));
//             std::vector<double> db2(outputSize);
//             for (int o = 0; o < outputSize; ++o)
//             {
//                 db2[o] = error[o];
//                 for (int h = 0; h < hiddenSize; ++h)
//                     dW2[o][h] = error[o] * a1[h];
//             }

//             std::vector<double> dA1(hiddenSize, 0.0);
//             for (int h = 0; h < hiddenSize; ++h)
//                 for (int o = 0; o < outputSize; ++o)
//                     dA1[h] += W2[o][h] * error[o];

//             std::vector<std::vector<double>> dW1(hiddenSize, std::vector<double>(inputSize));
//             std::vector<double> db1(hiddenSize);
//             for (int h = 0; h < hiddenSize; ++h)
//             {
//                 double dZ1 = dA1[h] * sigmoid_derivative(z1[h]);
//                 db1[h] = dZ1;
//                 for (int j = 0; j < inputSize; ++j)
//                     dW1[h][j] = dZ1 * x[j];
//             }

//             // ---- Update ----
//             add_to(W2, dW2, -learningRate);
//             add_to(b2, db2, -learningRate);
//             add_to(W1, dW1, -learningRate);
//             add_to(b1, db1, -learningRate);
//         }

//         // Calculate average MSE
//         avgMSE = totalMSE / n;

//         int correct = 0;
//         for (size_t i = 0; i < n; ++i)
//         {
//             int pred = predict(X[i]);
//             int actual = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
//             if (pred == actual)
//                 correct++;
//         }
//         double acc = 100.0 * correct / n;

//         mseHistory.push_back(avgMSE);
//         accuracyHistory.push_back(acc);

//         epoch++;
//     }
// }

void Model::train(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int epochs) {
  size_t n          = X.size();
  int    epoch      = 0;
  double avgMSE     = 1.0;
  int    iterations = 0;

  std::cout << "Starting training..." << std::endl;
  std::cout << "Epoch\t\tMSE\t\tAccuracy (%)" << std::endl;
  std::cout << "-----\t\t---\t\t-----------" << std::endl;

  while (avgMSE > 1e-6 && epoch < 5000) {
    double totalMSE = 0;

    for (size_t i = 0; i < n; ++i) {
      const auto &x = X[i];
      const auto &y = Y[i];

      // ---- Forward Pass ----
      // Layer 1
      std::vector<double> z1(hiddenSize1), a1(hiddenSize1);
      for (int h = 0; h < hiddenSize1; ++h) {
        z1[h] = dot(W1[h], x) + b1[h];
        a1[h] = sigmoid(z1[h]);
      }

      // Layer 2
      std::vector<double> z2(hiddenSize2), a2(hiddenSize2);
      for (int h = 0; h < hiddenSize2; ++h) {
        z2[h] = dot(W2[h], a1) + b2[h];
        a2[h] = sigmoid(z2[h]);
      }

      // Layer 3
      std::vector<double> z3(hiddenSize3), a3(hiddenSize3);
      for (int h = 0; h < hiddenSize3; ++h) {
        z3[h] = dot(W3[h], a2) + b3[h];
        a3[h] = sigmoid(z3[h]);
      }

      // Layer 4
      std::vector<double> z4(hiddenSize4), a4(hiddenSize4);
      for (int h = 0; h < hiddenSize4; ++h) {
        z4[h] = dot(W4[h], a3) + b4[h];
        a4[h] = sigmoid(z4[h]);
      }

      // Output layer
      std::vector<double> z5(outputSize), y_hat(outputSize);
      for (int o = 0; o < outputSize; ++o) z5[o] = dot(W5[o], a4) + b5[o];
      y_hat = softmax(z5);

      // ---- Loss ----
      double sampleMSE = 0.0;
      std::vector<double> error(outputSize);
      for (int o = 0; o < outputSize; ++o) {
        error[o] = y_hat[o] - y[o];
        sampleMSE += 0.5 * error[o] * error[o];
        totalMSE += 0.5 * error[o] * error[o];
      }

      // ---- Backpropagation ----
      // Output layer gradients
      std::vector<std::vector<double>> dW5(outputSize, std::vector<double>(hiddenSize4));
      std::vector<double>              db5(outputSize);
      for (int o = 0; o < outputSize; ++o) {
        db5[o] = error[o];
        for (int h = 0; h < hiddenSize4; ++h) dW5[o][h] = error[o] * a4[h];
      }

      // Hidden layer 4 gradients
      std::vector<double> dA4(hiddenSize4, 0.0);
      for (int h = 0; h < hiddenSize4; ++h)
        for (int o = 0; o < outputSize; ++o) dA4[h] += W5[o][h] * error[o];

      std::vector<std::vector<double>> dW4(hiddenSize4, std::vector<double>(hiddenSize3));
      std::vector<double>              db4(hiddenSize4);
      for (int h = 0; h < hiddenSize4; ++h) {
        double dZ4 = dA4[h] * sigmoid_derivative(z4[h]);
        db4[h]     = dZ4;
        for (int j = 0; j < hiddenSize3; ++j) dW4[h][j] = dZ4 * a3[j];
      }

      // Hidden layer 3 gradients - FIXED
      std::vector<double> dA3(hiddenSize3, 0.0);
      for (int h = 0; h < hiddenSize3; ++h)
        for (int k = 0; k < hiddenSize4; ++k) dA3[h] += W4[k][h] * db4[k];  // CORRECTED: use db4[k] instead of dA4[k] * sigmoid_derivative(z4[k])

      std::vector<std::vector<double>> dW3(hiddenSize3, std::vector<double>(hiddenSize2));
      std::vector<double>              db3(hiddenSize3);
      for (int h = 0; h < hiddenSize3; ++h) {
        double dZ3 = dA3[h] * sigmoid_derivative(z3[h]);
        db3[h]     = dZ3;
        for (int j = 0; j < hiddenSize2; ++j) dW3[h][j] = dZ3 * a2[j];
      }

      // Hidden layer 2 gradients
      std::vector<double> dA2(hiddenSize2, 0.0);
      for (int h = 0; h < hiddenSize2; ++h)
        for (int k = 0; k < hiddenSize3; ++k) dA2[h] += W3[k][h] * db3[k];  // CORRECT: use db3[k]

      std::vector<std::vector<double>> dW2(hiddenSize2, std::vector<double>(hiddenSize1));
      std::vector<double>              db2(hiddenSize2);
      for (int h = 0; h < hiddenSize2; ++h) {
        double dZ2 = dA2[h] * sigmoid_derivative(z2[h]);
        db2[h]     = dZ2;
        for (int j = 0; j < hiddenSize1; ++j) dW2[h][j] = dZ2 * a1[j];
      }

      // Hidden layer 1 gradients
      std::vector<double> dA1(hiddenSize1, 0.0);
      for (int h = 0; h < hiddenSize1; ++h)
        for (int k = 0; k < hiddenSize2; ++k) dA1[h] += W2[k][h] * db2[k];  // CORRECT: use db2[k]

      std::vector<std::vector<double>> dW1(hiddenSize1, std::vector<double>(inputSize));
      std::vector<double>              db1(hiddenSize1);
      for (int h = 0; h < hiddenSize1; ++h) {
        double dZ1 = dA1[h] * sigmoid_derivative(z1[h]);
        db1[h]     = dZ1;
        for (int j = 0; j < inputSize; ++j) dW1[h][j] = dZ1 * x[j];
      }

      // ---- Update Weights ----
      add_to(W5, dW5, -learningRate);
      add_to(b5, db5, -learningRate);
      add_to(W4, dW4, -learningRate);
      add_to(b4, db4, -learningRate);
      add_to(W3, dW3, -learningRate);
      add_to(b3, db3, -learningRate);
      add_to(W2, dW2, -learningRate);
      add_to(b2, db2, -learningRate);
      add_to(W1, dW1, -learningRate);
      add_to(b1, db1, -learningRate);

      
      if(iterations % 10000 == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Epoch: " << epoch << ", Sample: " << i << ", ";
        std::cout << "Sample MSE: " << sampleMSE << ", ";
        std::cout << "MSE: " << totalMSE / (i + 1) << ", ";
        std::cout << "Iteration: " << iterations << ", Sample MSE: " << sampleMSE << std::endl;
        mseHistory.push_back(sampleMSE);  // Store MSE for this sample
      }
      iterations++;  // Increment iteration count
    }
    
    // Calculate average MSE
    avgMSE = totalMSE / n;

    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
      int pred   = predict(X[i]);
      int actual = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
      if (pred == actual)
        correct++;
    }
    double acc = 100.0 * correct / n;

    // Store MSE and accuracy history per epoch
    // mseHistory.push_back(avgMSE);
    accuracyHistory.push_back(acc);

    // Print progress every 10 epochs or when target MSE is reached
    if (epoch % 10 == 0 || avgMSE <= 1e-6 || epoch == epochs - 1) {
      std::cout << std::fixed << std::setprecision(6);
      std::cout << epoch << "\t\t" << avgMSE << "\t" << std::setprecision(2) << acc << std::endl;
    }

    epoch++;

    // Early stopping if MSE target is reached
    if (avgMSE <= 1e-4) {
      std::cout << "\nTarget MSE (1e-6) reached at epoch " << epoch << std::endl;
      break;
    }
  }

  std::cout << "\nTraining completed!" << std::endl;
  std::cout << "Final MSE: " << std::scientific << avgMSE << std::endl;
  std::cout << "Total epochs: " << epoch << std::endl;
}

// std::vector<double> Model::predictProb(const std::vector<double> &input)
// {
//     std::vector<double> z1(hiddenSize), a1(hiddenSize);
//     for (int h = 0; h < hiddenSize; ++h)
//         a1[h] = sigmoid(dot(W1[h], input) + b1[h]);

//     std::vector<double> z2(outputSize);
//     for (int o = 0; o < outputSize; ++o)
//         z2[o] = dot(W2[o], a1) + b2[o];

//     return softmax(z2);
// }

std::vector<double> Model::predictProb(const std::vector<double> &input) {
  // Forward pass through all 4 hidden layers
  std::vector<double> a1(hiddenSize1);
  for (int h = 0; h < hiddenSize1; ++h) a1[h] = sigmoid(dot(W1[h], input) + b1[h]);

  std::vector<double> a2(hiddenSize2);
  for (int h = 0; h < hiddenSize2; ++h) a2[h] = sigmoid(dot(W2[h], a1) + b2[h]);

  std::vector<double> a3(hiddenSize3);
  for (int h = 0; h < hiddenSize3; ++h) a3[h] = sigmoid(dot(W3[h], a2) + b3[h]);

  std::vector<double> a4(hiddenSize4);
  for (int h = 0; h < hiddenSize4; ++h) a4[h] = sigmoid(dot(W4[h], a3) + b4[h]);

  std::vector<double> z5(outputSize);
  for (int o = 0; o < outputSize; ++o) z5[o] = dot(W5[o], a4) + b5[o];

  return softmax(z5);
}

int Model::predict(const std::vector<double> &input) {
  auto probs = predictProb(input);
  return std::max_element(probs.begin(), probs.end()) - probs.begin();
}

std::vector<double> Model::getMSEHistory() const {
  return mseHistory;
}

std::vector<double> Model::getAccuracyHistory() const {
  return accuracyHistory;
}

void Model::normalizeTrain(std::vector<std::vector<double>> &X, double &rrMin, double &rrMax, double &qrsMin, double &qrsMax) {
  rrMin = qrsMin = 1e9;
  rrMax = qrsMax = -1e9;

  for (const auto &x : X) {
    rrMin  = std::min(rrMin, x[0]);
    rrMax  = std::max(rrMax, x[0]);
    qrsMin = std::min(qrsMin, x[1]);
    qrsMax = std::max(qrsMax, x[1]);
  }

  for (auto &x : X) {
    x[0] = (x[0] - rrMin) / (rrMax - rrMin + 1e-9);
    x[1] = (x[1] - qrsMin) / (qrsMax - qrsMin + 1e-9);
  }
}

int Model::classifyArrhythmia(double rr, double qrsd) {
  if (rr >= 0.6 && rr <= 1.2 && qrsd >= 80 && qrsd <= 100)
    return 0;  // Normal
  else if (rr < 0.32)
    return 3;  // Tachycardia
  else if (rr >= 0.32 && rr < 0.60 && qrsd >= 80 && qrsd <= 100)
    return 2;  // R-on-T
  else if (rr >= 0.32 && rr < 0.60 && qrsd > 100)
    return 6;  // PVC
  else if (rr >= 0.6 && rr <= 1.2 && qrsd > 100)
    return 5;  // Fusion
  else if (rr > 1.2 && rr <= 1.66)
    return 4;  // Bradycardia
  else if (rr > 1.66)
    return 1;  // Dropped
  return -1;
}