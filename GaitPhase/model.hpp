#ifndef MODEL_HPP
#define MODEL_HPP

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x) {
  return x * (1.0 - x);  // untuk output dari sigmoid, bukan input
}

class MLP {
 private:
  static constexpr int INPUT_SIZE  = 2;
  static constexpr int HIDDEN_SIZE = 16;
  static constexpr int OUTPUT_SIZE = 1;

  // Bobot dan bias
  double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
  double bias_hidden[HIDDEN_SIZE];
  double weights_hidden_output[HIDDEN_SIZE];
  double bias_output;

  // Aktivasi
  double hidden[HIDDEN_SIZE];
  double output;

 public:
  MLP() {
    std::srand(std::time(0));
    for (int i = 0; i < INPUT_SIZE; ++i)
      for (int j = 0; j < HIDDEN_SIZE; ++j) weights_input_hidden[i][j] = ((double) rand() / RAND_MAX - 0.5);

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      bias_hidden[j]           = ((double) rand() / RAND_MAX - 0.5);
      weights_hidden_output[j] = ((double) rand() / RAND_MAX - 0.5);
    }

    bias_output = ((double) rand() / RAND_MAX - 0.5);
  }

  double predict(double x1, double x2) {
    // Feedforward
    for (int j = 0; j < HIDDEN_SIZE; ++j) {
      hidden[j] = sigmoid(x1 * weights_input_hidden[0][j] + x2 * weights_input_hidden[1][j] + bias_hidden[j]);
    }

    double sum = bias_output;
    for (int j = 0; j < HIDDEN_SIZE; ++j) sum += hidden[j] * weights_hidden_output[j];

    output = sigmoid(sum);
    return output;
  }

  void train(const std::vector<std::pair<std::vector<double>, int>>& dataset, int epochs = 1000, double lr = 0.1) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      for (const auto& sample : dataset) {
        double x1     = sample.first[0];
        double x2     = sample.first[1];
        int    target = sample.second;

        // Feedforward
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
          hidden[j] = sigmoid(x1 * weights_input_hidden[0][j] + x2 * weights_input_hidden[1][j] + bias_hidden[j]);
        }

        double sum = bias_output;
        for (int j = 0; j < HIDDEN_SIZE; ++j) sum += hidden[j] * weights_hidden_output[j];

        output = sigmoid(sum);

        // Backpropagation
        double error_output = (target - output) * sigmoid_derivative(output);

        double error_hidden[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
          error_hidden[j] = error_output * weights_hidden_output[j] * sigmoid_derivative(hidden[j]);
        }

        // Update weights dan bias
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
          weights_hidden_output[j] += lr * error_output * hidden[j];
        }
        bias_output += lr * error_output;

        for (int i = 0; i < INPUT_SIZE; ++i) {
          for (int j = 0; j < HIDDEN_SIZE; ++j) {
            double input = (i == 0) ? x1 : x2;
            weights_input_hidden[i][j] += lr * error_hidden[j] * input;
          }
        }

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
          bias_hidden[j] += lr * error_hidden[j];
        }
      }
    }
  }
};

#endif
