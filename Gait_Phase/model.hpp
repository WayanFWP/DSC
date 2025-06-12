#ifndef MODEL_HPP
#define MODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class MLP {
 private:
  static const int INPUT_SIZE   = 2;
  static const int HIDDEN1_SIZE = 8;
  static const int HIDDEN2_SIZE = 6;
  static const int OUTPUT_SIZE  = 6;

  std::vector<std::vector<double>> W1;     // [HIDDEN1_SIZE][INPUT_SIZE]
  std::vector<double>              bias1;  // [HIDDEN1_SIZE]

  std::vector<std::vector<double>> W2;     // [HIDDEN2_SIZE][HIDDEN1_SIZE]
  std::vector<double>              bias2;  // [HIDDEN2_SIZE]

  std::vector<std::vector<double>> W3;     // [OUTPUT_SIZE][HIDDEN2_SIZE]
  std::vector<double>              bias3;  // [OUTPUT_SIZE]

  std::vector<std::vector<double>> mW1, vW1;
  std::vector<double>              mbias1, vbias1;

  std::vector<std::vector<double>> mW2, vW2;
  std::vector<double>              mbias2, vbias2;

  std::vector<std::vector<double>> mW3, vW3;
  std::vector<double>              mbias3, vbias3;

  int    timestep;
  double beta1;
  double beta2;
  double epsilon;

  // Activations
  std::vector<double> hidden1;
  std::vector<double> hidden2;
  std::vector<double> output;
  double              sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  double              sigmoid_derivative(double x) { return x * (1.0 - x); }

  void softmax(const std::vector<double>& input, std::vector<double>& output) {
    double max_val = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
      if (input[i] > max_val)
        max_val = input[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = std::exp(input[i] - max_val);
      sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
      output[i] /= sum;
    }
  }
  double calculateMSE(const std::vector<double>& output, int target) {
    double mse = 0.0;
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
      double target_val = (o == target) ? 1.0 : 0.0;
      double error      = output[o] - target_val;
      mse += error * error;
    }
    return mse / (2.0 * OUTPUT_SIZE);
  }

 public:
  MLP(double learning_rate = 0.1, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8) : beta1(b1), beta2(b2), epsilon(eps), timestep(0) {
    std::random_device rd;
    std::mt19937       gen(rd());

    // Initialize with proper dimensions
    W1.resize(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
    bias1.resize(HIDDEN1_SIZE);

    W2.resize(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE));
    bias2.resize(HIDDEN2_SIZE);

    W3.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN2_SIZE));
    bias3.resize(OUTPUT_SIZE);

    // Use Xavier/Glorot initialization for better gradient flow
    double w1_limit = std::sqrt(6.0 / (INPUT_SIZE + HIDDEN1_SIZE));
    double w2_limit = std::sqrt(6.0 / (HIDDEN1_SIZE + HIDDEN2_SIZE));
    double w3_limit = std::sqrt(6.0 / (HIDDEN2_SIZE + OUTPUT_SIZE));

    // Initialize Adam optimizer states
    mW1.resize(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
    vW1.resize(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
    mbias1.resize(HIDDEN1_SIZE, 0.0);
    vbias1.resize(HIDDEN1_SIZE, 0.0);

    mW2.resize(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE, 0.0));
    vW2.resize(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE, 0.0));
    mbias2.resize(HIDDEN2_SIZE, 0.0);
    vbias2.resize(HIDDEN2_SIZE, 0.0);

    mW3.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN2_SIZE, 0.0));
    vW3.resize(OUTPUT_SIZE, std::vector<double>(HIDDEN2_SIZE, 0.0));
    mbias3.resize(OUTPUT_SIZE, 0.0);
    vbias3.resize(OUTPUT_SIZE, 0.0);

    std::uniform_real_distribution<double> dist1(-w1_limit, w1_limit);
    std::uniform_real_distribution<double> dist2(-w2_limit, w2_limit);
    std::uniform_real_distribution<double> dist3(-w3_limit, w3_limit);
    std::uniform_real_distribution<double> dist_bias(-0.1, 0.1);

    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
      bias1[i] = dist_bias(gen);
      for (int j = 0; j < INPUT_SIZE; ++j) {
        W1[i][j] = dist1(gen);
      }
    }

    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
      bias2[i] = dist_bias(gen);
      for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        W2[i][j] = dist2(gen);
      }
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
      bias3[i] = dist_bias(gen);
      for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        W3[i][j] = dist3(gen);
      }
    }

    // Initialize activations
    hidden1.resize(HIDDEN1_SIZE);
    hidden2.resize(HIDDEN2_SIZE);
    output.resize(OUTPUT_SIZE);
  }
  double min_x1 = 0.0, max_x1 = 2.0;
  double min_x2 = 0.0, max_x2 = 2.0;

  void setNormalizationParams(double min1, double max1, double min2, double max2) {
    min_x1 = min1;
    max_x1 = max1;
    min_x2 = min2;
    max_x2 = max2;
  }

  std::vector<double> predict(double x1, double x2) {
    // Normalize inputs the same way as in training
    double norm_x1 = (max_x1 > min_x1) ? (x1 - min_x1) / (max_x1 - min_x1) : x1;
    double norm_x2 = (max_x2 > min_x2) ? (x2 - min_x2) / (max_x2 - min_x2) : x2;

    std::vector<double> input = {norm_x1, norm_x2};

    // Forward pass
    // Hidden layer 1
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
      double z = bias1[i];
      for (int j = 0; j < INPUT_SIZE; ++j) {
        z += W1[i][j] * input[j];
      }
      hidden1[i] = sigmoid(z);
    }

    // Hidden layer 2
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
      double z = bias2[i];
      for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        z += W2[i][j] * hidden1[j];
      }
      hidden2[i] = sigmoid(z);
    }

    // Output layer
    std::vector<double> output_pre_softmax(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
      double z = bias3[i];
      for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        z += W3[i][j] * hidden2[j];
      }
      output_pre_softmax[i] = z;
    }

    // Apply softmax
    softmax(output_pre_softmax, output);

    return output;
  }
  void train(const std::vector<std::pair<std::vector<double>, int>>& dataset, int epochs = 1000,
             double learning_rate = 0.25, 
             double decay_rate    = 0.01) {
    std::cout << "Training on dataset with " << dataset.size() << " samples" << std::endl;
    std::cout << "Initial learning rate: " << learning_rate << std::endl;

    // Find data ranges for simple normalization
    double min_x1 = 999999, max_x1 = -999999;
    double min_x2 = 999999, max_x2 = -999999;
    for (const auto& sample : dataset) {
      min_x1 = std::min(min_x1, sample.first[0]);
      max_x1 = std::max(max_x1, sample.first[0]);
      min_x2 = std::min(min_x2, sample.first[1]);
      max_x2 = std::max(max_x2, sample.first[1]);
    }
    std::cout << "Data ranges: x1 [" << min_x1 << ", " << max_x1 << "], x2 [" << min_x2 << ", " << max_x2 << "]" << std::endl;

    // Store normalization parameters for later prediction
    setNormalizationParams(min_x1, max_x1, min_x2, max_x2);

    std::ofstream log_file("training_log.txt");
    double        prev_loss = 999999.0;  // For tracking loss change

    // Initialize random shuffle of dataset indices
    std::vector<int> indices(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
      indices[i] = static_cast<int>(i);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
      // Calculate adaptive learning rate with decay
      double lr = learning_rate / (1.0 + decay_rate * epoch);

      double total_loss = 0.0;
      // Shuffle indices for stochastic training using C++11 method
      std::random_device rd;
      std::mt19937       g(rd());
      std::shuffle(indices.begin(), indices.end(), g);
      for (size_t idx = 0; idx < dataset.size(); ++idx) {
        const auto&         sample = dataset[indices[idx]];
        std::vector<double> input  = sample.first;

        // Simple normalization to improve training
        if (max_x1 > min_x1)
          input[0] = (input[0] - min_x1) / (max_x1 - min_x1);
        if (max_x2 > min_x2)
          input[1] = (input[1] - min_x2) / (max_x2 - min_x2);

        int target = sample.second;

        // Forward pass
        // Hidden layer 1
        for (int i = 0; i < HIDDEN1_SIZE; ++i) {
          double z = bias1[i];
          for (int j = 0; j < INPUT_SIZE; ++j) {
            z += W1[i][j] * input[j];
          }
          hidden1[i] = sigmoid(z);
        }

        // Hidden layer 2
        for (int i = 0; i < HIDDEN2_SIZE; ++i) {
          double z = bias2[i];
          for (int j = 0; j < HIDDEN1_SIZE; ++j) {
            z += W2[i][j] * hidden1[j];
          }
          hidden2[i] = sigmoid(z);
        }

        // Output layer
        std::vector<double> output_pre_softmax(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
          double z = bias3[i];
          for (int j = 0; j < HIDDEN2_SIZE; ++j) {
            z += W3[i][j] * hidden2[j];
          }
          output_pre_softmax[i] = z;
        }

        // Apply softmax
        softmax(output_pre_softmax, output);

        // Calculate loss
        total_loss += calculateMSE(output, target);
        // Backward pass
        std::vector<double> d_output(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
          double target_val = (i == target) ? 1.0 : 0.0;
          // MSE derivative with respect to softmax output
          d_output[i] = (output[i] - target_val);
        }

        // Output layer gradients
        std::vector<std::vector<double>> dW3(OUTPUT_SIZE, std::vector<double>(HIDDEN2_SIZE, 0.0));
        std::vector<double>              db3(OUTPUT_SIZE, 0.0);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
          for (int j = 0; j < HIDDEN2_SIZE; ++j) {
            dW3[i][j] = d_output[i] * hidden2[j];
          }
          db3[i] = d_output[i];
        }

        // Hidden layer 2 gradients
        std::vector<double> d_h2(HIDDEN2_SIZE, 0.0);
        for (int j = 0; j < HIDDEN2_SIZE; ++j) {
          double error = 0.0;
          for (int i = 0; i < OUTPUT_SIZE; ++i) {
            error += W3[i][j] * d_output[i];
          }
          d_h2[j] = error * sigmoid_derivative(hidden2[j]);
        }

        std::vector<std::vector<double>> dW2(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE, 0.0));
        std::vector<double>              db2(HIDDEN2_SIZE, 0.0);
        for (int i = 0; i < HIDDEN2_SIZE; ++i) {
          for (int j = 0; j < HIDDEN1_SIZE; ++j) {
            dW2[i][j] = d_h2[i] * hidden1[j];
          }
          db2[i] = d_h2[i];
        }

        // Hidden layer 1 gradients
        std::vector<double> d_h1(HIDDEN1_SIZE, 0.0);
        for (int j = 0; j < HIDDEN1_SIZE; ++j) {
          double error = 0.0;
          for (int i = 0; i < HIDDEN2_SIZE; ++i) {
            error += W2[i][j] * d_h2[i];
          }
          d_h1[j] = error * sigmoid_derivative(hidden1[j]);
        }

        std::vector<std::vector<double>> dW1(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE, 0.0));
        std::vector<double>              db1(HIDDEN1_SIZE, 0.0);
        for (int i = 0; i < HIDDEN1_SIZE; ++i) {
          for (int j = 0; j < INPUT_SIZE; ++j) {
            dW1[i][j] = d_h1[i] * input[j];
          }
          db1[i] = d_h1[i];
        }

        // Update parameters using Adam optimizer
        timestep++;
        updateParameters(dW1, db1, dW2, db2, dW3, db3, lr);
      }
      // Log the loss for this epoch
      double avg_loss = total_loss / dataset.size();
      if (epoch % 10 == 0 || epoch == epochs - 1) {
        double loss_change = prev_loss - avg_loss;
        log_file << "Epoch " << epoch + 1 << ": MSE = " << avg_loss << std::endl;
        std::cout << "Epoch " << epoch + 1 << ": MSE = " << avg_loss << " (lr = " << lr << ", change: " << loss_change << ")" << std::endl;

        // Check for nearly zero change (convergence)
        if (std::abs(loss_change) < 1e-6 && epoch > 50) {
          std::cout << "Training converged with very small change in loss." << std::endl;
          break;
        }

        // Early stopping if loss increases
        if (loss_change < 0 && epoch > 100) {
          learning_rate *= 0.5;  // Reduce learning rate if loss increases
          std::cout << "Loss increased, reducing learning rate to " << learning_rate << std::endl;
          if (learning_rate < 1e-5) {
            std::cout << "Learning rate too small, stopping training." << std::endl;
            break;
          }
        }

        prev_loss = avg_loss;
      }
    }

    log_file.close();
  }
  void updateParameters(const std::vector<std::vector<double>>& dW1, const std::vector<double>& db1, const std::vector<std::vector<double>>& dW2,
                        const std::vector<double>& db2, const std::vector<std::vector<double>>& dW3, const std::vector<double>& db3, double lr) {
    // For debugging - track gradient statistics
    double max_grad   = 0.0;
    double avg_grad   = 0.0;
    int    grad_count = 0;
    // Update weights and biases using Adam optimizer
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
      for (int j = 0; j < INPUT_SIZE; ++j) {
        // Track gradient statistics
        double abs_grad = std::abs(dW1[i][j]);
        if (abs_grad > max_grad)
          max_grad = abs_grad;
        avg_grad += abs_grad;
        grad_count++;

        // Apply gradient clipping to prevent explosion
        double clipped_grad = std::max(std::min(dW1[i][j], 1.0), -1.0);

        mW1[i][j]    = beta1 * mW1[i][j] + (1 - beta1) * clipped_grad;
        vW1[i][j]    = beta2 * vW1[i][j] + (1 - beta2) * clipped_grad * clipped_grad;
        double m_hat = mW1[i][j] / (1 - std::pow(beta1, timestep));
        double v_hat = vW1[i][j] / (1 - std::pow(beta2, timestep));
        W1[i][j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      // Track bias gradients too
      double abs_grad = std::abs(db1[i]);
      if (abs_grad > max_grad)
        max_grad = abs_grad;
      avg_grad += abs_grad;
      grad_count++;

      // Apply gradient clipping
      double clipped_grad = std::max(std::min(db1[i], 1.0), -1.0);

      mbias1[i]    = beta1 * mbias1[i] + (1 - beta1) * clipped_grad;
      vbias1[i]    = beta2 * vbias1[i] + (1 - beta2) * clipped_grad * clipped_grad;
      double m_hat = mbias1[i] / (1 - std::pow(beta1, timestep));
      double v_hat = vbias1[i] / (1 - std::pow(beta2, timestep));
      bias1[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    // Display gradient info every 100 iterations
    if (timestep % 100 == 0) {
      avg_grad = grad_count > 0 ? avg_grad / grad_count : 0;
      std::cout << "Timestep: " << timestep << " | Max grad: " << max_grad << " | Avg grad: " << avg_grad << std::endl;
    }

    // Handle hidden layer 2 parameters
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
      for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        // Track and clip gradients
        double abs_grad = std::abs(dW2[i][j]);
        if (abs_grad > max_grad)
          max_grad = abs_grad;
        avg_grad += abs_grad;
        grad_count++;

        double clipped_grad = std::max(std::min(dW2[i][j], 1.0), -1.0);

        mW2[i][j]    = beta1 * mW2[i][j] + (1 - beta1) * clipped_grad;
        vW2[i][j]    = beta2 * vW2[i][j] + (1 - beta2) * clipped_grad * clipped_grad;
        double m_hat = mW2[i][j] / (1 - std::pow(beta1, timestep));
        double v_hat = vW2[i][j] / (1 - std::pow(beta2, timestep));
        W2[i][j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      // Track and clip bias gradients
      double abs_grad = std::abs(db2[i]);
      if (abs_grad > max_grad)
        max_grad = abs_grad;
      avg_grad += abs_grad;
      grad_count++;

      double clipped_grad = std::max(std::min(db2[i], 1.0), -1.0);

      mbias2[i]    = beta1 * mbias2[i] + (1 - beta1) * clipped_grad;
      vbias2[i]    = beta2 * vbias2[i] + (1 - beta2) * clipped_grad * clipped_grad;
      double m_hat = mbias2[i] / (1 - std::pow(beta1, timestep));
      double v_hat = vbias2[i] / (1 - std::pow(beta2, timestep));
      bias2[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
      for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        // Track and clip gradients
        double abs_grad = std::abs(dW3[i][j]);
        if (abs_grad > max_grad)
          max_grad = abs_grad;
        avg_grad += abs_grad;
        grad_count++;

        double clipped_grad = std::max(std::min(dW3[i][j], 1.0), -1.0);

        mW3[i][j]    = beta1 * mW3[i][j] + (1 - beta1) * clipped_grad;
        vW3[i][j]    = beta2 * vW3[i][j] + (1 - beta2) * clipped_grad * clipped_grad;
        double m_hat = mW3[i][j] / (1 - std::pow(beta1, timestep));
        double v_hat = vW3[i][j] / (1 - std::pow(beta2, timestep));
        W3[i][j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      // Track and clip bias gradients
      double abs_grad = std::abs(db3[i]);
      if (abs_grad > max_grad)
        max_grad = abs_grad;
      avg_grad += abs_grad;
      grad_count++;

      double clipped_grad = std::max(std::min(db3[i], 1.0), -1.0);

      mbias3[i]    = beta1 * mbias3[i] + (1 - beta1) * clipped_grad;
      vbias3[i]    = beta2 * vbias3[i] + (1 - beta2) * clipped_grad * clipped_grad;
      double m_hat = mbias3[i] / (1 - std::pow(beta1, timestep));
      double v_hat = vbias3[i] / (1 - std::pow(beta2, timestep));
      bias3[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
  }
};

#endif