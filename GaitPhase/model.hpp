#ifndef MODEL_HPP
#define MODEL_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

class MLP {
 private:
  static constexpr int INPUT_SIZE   = 2;
  static constexpr int HIDDEN1_SIZE = 8;
  static constexpr int HIDDEN2_SIZE = 6;
  static constexpr int OUTPUT_SIZE  = 1;

  // Weights & biases
  double weights_input_hidden1[INPUT_SIZE][HIDDEN1_SIZE];
  double bias_hidden1[HIDDEN1_SIZE];

  double weights_hidden1_hidden2[HIDDEN1_SIZE][HIDDEN2_SIZE];
  double bias_hidden2[HIDDEN2_SIZE];

  double weights_hidden2_output[HIDDEN2_SIZE];
  double bias_output;

  // Activations
  double hidden1[HIDDEN1_SIZE];
  double hidden2[HIDDEN2_SIZE];
  double output;

  double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  double sigmoid_derivative(double x) { return x * (1.0 - x); }

 public:
  MLP() {
    std::srand(std::time(0));

    for (int i = 0; i < INPUT_SIZE; ++i)
      for (int j = 0; j < HIDDEN1_SIZE; ++j) weights_input_hidden1[i][j] = ((double) rand() / RAND_MAX - 0.5);

    for (int j = 0; j < HIDDEN1_SIZE; ++j) {
      bias_hidden1[j] = ((double) rand() / RAND_MAX - 0.5);
      for (int k = 0; k < HIDDEN2_SIZE; ++k) weights_hidden1_hidden2[j][k] = ((double) rand() / RAND_MAX - 0.5);
    }

    for (int k = 0; k < HIDDEN2_SIZE; ++k) {
      bias_hidden2[k]           = ((double) rand() / RAND_MAX - 0.5);
      weights_hidden2_output[k] = ((double) rand() / RAND_MAX - 0.5);
    }

    bias_output = ((double) rand() / RAND_MAX - 0.5);
  }

  double predict(double x1, double x2) {
    for (int j = 0; j < HIDDEN1_SIZE; ++j)
      hidden1[j] = sigmoid(x1 * weights_input_hidden1[0][j] + x2 * weights_input_hidden1[1][j] + bias_hidden1[j]);

    for (int k = 0; k < HIDDEN2_SIZE; ++k) {
      double sum = bias_hidden2[k];
      for (int j = 0; j < HIDDEN1_SIZE; ++j) sum += hidden1[j] * weights_hidden1_hidden2[j][k];
      hidden2[k] = sigmoid(sum);
    }

    double sum_out = bias_output;
    for (int k = 0; k < HIDDEN2_SIZE; ++k) sum_out += hidden2[k] * weights_hidden2_output[k];

    output = sigmoid(sum_out);
    return output;
  }

  void train(const std::vector<std::pair<std::vector<double>, int>>& dataset, int epochs = 1000, double lr = 0.1) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      for (const auto& sample : dataset) {
        double x1     = sample.first[0];
        double x2     = sample.first[1];
        int    target = sample.second;

        // Feedforward
        for (int j = 0; j < HIDDEN1_SIZE; ++j)
          hidden1[j] = sigmoid(x1 * weights_input_hidden1[0][j] + x2 * weights_input_hidden1[1][j] + bias_hidden1[j]);

        for (int k = 0; k < HIDDEN2_SIZE; ++k) {
          double sum = bias_hidden2[k];
          for (int j = 0; j < HIDDEN1_SIZE; ++j) sum += hidden1[j] * weights_hidden1_hidden2[j][k];
          hidden2[k] = sigmoid(sum);
        }

        double sum_out = bias_output;
        for (int k = 0; k < HIDDEN2_SIZE; ++k) sum_out += hidden2[k] * weights_hidden2_output[k];

        output = sigmoid(sum_out);

        // Backpropagation
        double error_output = (target - output) * sigmoid_derivative(output);
        double error_hidden2[HIDDEN2_SIZE];
        for (int k = 0; k < HIDDEN2_SIZE; ++k) error_hidden2[k] = error_output * weights_hidden2_output[k] * sigmoid_derivative(hidden2[k]);

        double error_hidden1[HIDDEN1_SIZE];
        for (int j = 0; j < HIDDEN1_SIZE; ++j) {
          error_hidden1[j] = 0.0;
          for (int k = 0; k < HIDDEN2_SIZE; ++k) error_hidden1[j] += error_hidden2[k] * weights_hidden1_hidden2[j][k];
          error_hidden1[j] *= sigmoid_derivative(hidden1[j]);
        }

        for (int k = 0; k < HIDDEN2_SIZE; ++k) weights_hidden2_output[k] += lr * error_output * hidden2[k];
        bias_output += lr * error_output;

        for (int j = 0; j < HIDDEN1_SIZE; ++j)
          for (int k = 0; k < HIDDEN2_SIZE; ++k) weights_hidden1_hidden2[j][k] += lr * error_hidden2[k] * hidden1[j];
        for (int k = 0; k < HIDDEN2_SIZE; ++k) bias_hidden2[k] += lr * error_hidden2[k];

        for (int i = 0; i < INPUT_SIZE; ++i) {
          double input = (i == 0) ? x1 : x2;
          for (int j = 0; j < HIDDEN1_SIZE; ++j) weights_input_hidden1[i][j] += lr * error_hidden1[j] * input;
        }
        for (int j = 0; j < HIDDEN1_SIZE; ++j) bias_hidden1[j] += lr * error_hidden1[j];
      }
    }
  }
};

#endif
