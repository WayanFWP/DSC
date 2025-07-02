#ifndef MODEL_HPP
#define MODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class MLP {
 private:
  std::vector<double> loss_history;
  std::vector<double> accuracy_history;

  static const int INPUT_SIZE   = 2;
  static const int HIDDEN_SIZE  = 8;  // Single hidden layer like Aritmia
  static const int OUTPUT_SIZE  = 6;

  // Simplified architecture like Aritmia model
  std::vector<std::vector<double>> W1;     // [HIDDEN_SIZE][INPUT_SIZE]
  std::vector<double>              b1;     // [HIDDEN_SIZE]
  std::vector<std::vector<double>> W2;     // [OUTPUT_SIZE][HIDDEN_SIZE]
  std::vector<double>              b2;     // [OUTPUT_SIZE]

  double learning_rate;

  // Utility functions
  double sigmoid(double x);
  double sigmoid_derivative(double x);
  std::vector<double> softmax(const std::vector<double>& input);
  double dot(const std::vector<double>& a, const std::vector<double>& b);
  void add_to(std::vector<double>& a, const std::vector<double>& b, double scale = 1.0);
  void add_to(std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, double scale = 1.0);

 public:
  MLP(double lr = 0.01);

  // Qt purpose callbacks
  using ProgressCallback = std::function<void(int epoch, double loss, double lr, double loss_change)>;
  ProgressCallback progress_callback = nullptr;
  const std::vector<double>& getLossHistory() const { return loss_history; }
  const std::vector<double>& getAccuracyHistory() const { return accuracy_history; }
  void setProgressCallback(ProgressCallback callback) { progress_callback = callback; }

  // Normalization parameters
  double min_x1 = 0.0, max_x1 = 2.0;
  double min_x2 = 0.0, max_x2 = 2.0;
  void setNormalizationParams(double min1, double max1, double min2, double max2);
  void normalizeTrainingData(std::vector<std::vector<double>>& X);

  std::vector<double> predict(double x1, double x2);
  std::vector<double> predictProb(const std::vector<double>& input);
  int predictClass(const std::vector<double>& input);

  void train(const std::vector<std::pair<std::vector<double>, int>>& dataset, int epochs = 1000);
};

#endif