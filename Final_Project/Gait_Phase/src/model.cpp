#include "model.hpp"

MLP::MLP(double lr) : learning_rate(lr) {
  std::srand(std::time(0));

  // Initialize weights and biases for 3 hidden layers
  W1 = std::vector<std::vector<double>>(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
  b1 = std::vector<double>(HIDDEN1_SIZE, 0.0);
  W2 = std::vector<std::vector<double>>(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE));
  b2 = std::vector<double>(HIDDEN2_SIZE, 0.0);
  W3 = std::vector<std::vector<double>>(HIDDEN3_SIZE, std::vector<double>(HIDDEN2_SIZE));
  b3 = std::vector<double>(HIDDEN3_SIZE, 0.0);
  W4 = std::vector<std::vector<double>>(OUTPUT_SIZE, std::vector<double>(HIDDEN3_SIZE));
  b4 = std::vector<double>(OUTPUT_SIZE, 0.0);

  // Xavier initialization for each layer
  double std_w1 = sqrt(2.0 / INPUT_SIZE);
  double std_w2 = sqrt(2.0 / HIDDEN1_SIZE);
  double std_w3 = sqrt(2.0 / HIDDEN2_SIZE);
  double std_w4 = sqrt(2.0 / HIDDEN3_SIZE);

  std::random_device               rd;
  std::mt19937                     gen(rd());
  std::normal_distribution<double> dist1(0.0, std_w1);
  std::normal_distribution<double> dist2(0.0, std_w2);
  std::normal_distribution<double> dist3(0.0, std_w3);
  std::normal_distribution<double> dist4(0.0, std_w4);

  // Initialize W1
  for (auto& row : W1) {
    for (auto& val : row) {
      val = dist1(gen);
    }
  }

  // Initialize W2
  for (auto& row : W2) {
    for (auto& val : row) {
      val = dist2(gen);
    }
  }

  // Initialize W3
  for (auto& row : W3) {
    for (auto& val : row) {
      val = dist3(gen);
    }
  }

  // Initialize W4
  for (auto& row : W4) {
    for (auto& val : row) {
      val = dist4(gen);
    }
  }
}

double MLP::sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

std::vector<double> MLP::softmax(const std::vector<double>& input) {
  double              max_val = *std::max_element(input.begin(), input.end());
  std::vector<double> result(input.size());
  double              sum = 0.0;

  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = std::exp(input[i] - max_val);
    sum += result[i];
  }

  for (auto& val : result) {
    val /= sum;
  }

  return result;
}

double MLP::dot(const std::vector<double>& a, const std::vector<double>& b) {
  double result = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

void MLP::add_to(std::vector<double>& a, const std::vector<double>& b, double scale) {
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] += b[i] * scale;
  }
}

void MLP::add_to(std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, double scale) {
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < a[i].size(); ++j) {
      a[i][j] += b[i][j] * scale;
    }
  }
}

void MLP::setNormalizationParams(double min1, double max1, double min2, double max2) {
  min_x1 = min1;
  max_x1 = max1;
  min_x2 = min2;
  max_x2 = max2;
}

void MLP::normalizeTrainingData(std::vector<std::vector<double>>& X) {
  if (X.empty())
    return;

  min_x1 = min_x2 = 1e9;
  max_x1 = max_x2 = -1e9;

  // Find min/max values
  for (const auto& x : X) {
    min_x1 = std::min(min_x1, x[0]);
    max_x1 = std::max(max_x1, x[0]);
    min_x2 = std::min(min_x2, x[1]);
    max_x2 = std::max(max_x2, x[1]);
  }

  // Add small epsilon to avoid division by zero
  double eps      = 1e-8;
  double range_x1 = max_x1 - min_x1 + eps;
  double range_x2 = max_x2 - min_x2 + eps;

  // Normalize data to [0, 1] range
  for (auto& x : X) {
    x[0] = (x[0] - min_x1) / range_x1;
    x[1] = (x[1] - min_x2) / range_x2;
  }

  std::cout << "Normalization params - X1: [" << min_x1 << ", " << max_x1 << "], X2: [" << min_x2 << ", " << max_x2 << "]" << std::endl;
}

std::vector<double> MLP::predictProb(const std::vector<double>& input) {
  // Forward pass through 3 hidden layers
  
  // Hidden layer 1
  std::vector<double> z1(HIDDEN1_SIZE), a1(HIDDEN1_SIZE);
  for (int h = 0; h < HIDDEN1_SIZE; ++h) {
    z1[h] = dot(W1[h], input) + b1[h];
    a1[h] = sigmoid(z1[h]);
  }

  // Hidden layer 2
  std::vector<double> z2(HIDDEN2_SIZE), a2(HIDDEN2_SIZE);
  for (int h = 0; h < HIDDEN2_SIZE; ++h) {
    z2[h] = dot(W2[h], a1) + b2[h];
    a2[h] = sigmoid(z2[h]);
  }

  // Hidden layer 3
  std::vector<double> z3(HIDDEN3_SIZE), a3(HIDDEN3_SIZE);
  for (int h = 0; h < HIDDEN3_SIZE; ++h) {
    z3[h] = dot(W3[h], a2) + b3[h];
    a3[h] = sigmoid(z3[h]);
  }

  // Output layer
  std::vector<double> z4(OUTPUT_SIZE);
  for (int o = 0; o < OUTPUT_SIZE; ++o) {
    z4[o] = dot(W4[o], a3) + b4[o];
  }

  return softmax(z4);
}

int MLP::predictClass(const std::vector<double>& input) {
  auto probs = predictProb(input);
  return std::max_element(probs.begin(), probs.end()) - probs.begin();
}

std::vector<double> MLP::predict(double x1, double x2) {
  // Normalize inputs
  double norm_x1 = (max_x1 > min_x1) ? (x1 - min_x1) / (max_x1 - min_x1) : x1;
  double norm_x2 = (max_x2 > min_x2) ? (x2 - min_x2) / (max_x2 - min_x2) : x2;

  std::vector<double> input = {norm_x1, norm_x2};
  return predictProb(input);
}

void MLP::train(const std::vector<std::pair<std::vector<double>, int>>& dataset, int epochs) {
  std::cout << "Training model on " << dataset.size() << " samples" << std::endl;

  if (dataset.size() < 10) {
    std::cout << "WARNING: Very small dataset! Consider data augmentation." << std::endl;
  }

  loss_history.clear();
  accuracy_history.clear();

  // Prepare training data with feature engineering
  std::vector<std::vector<double>> X;
  std::vector<std::vector<double>> Y;

  for (const auto& sample : dataset) {
    // Original features
    double heel = sample.first[0];
    double toe  = sample.first[1];

    // Feature engineering
    std::vector<double> features;
    features.push_back(heel);                 // Raw heel
    features.push_back(toe);                  // Raw toe
    features.push_back(heel + toe);           // Total pressure
    features.push_back(fabs(heel - toe));     // Pressure difference
    features.push_back(heel / (toe + 1e-8));  // Heel/toe ratio
    features.push_back(heel * toe);           // Pressure product
    features.push_back(heel * heel);          // Heel squared
    features.push_back(toe * toe);            // Toe squared

    X.push_back(features);

    std::vector<double> y_onehot(OUTPUT_SIZE, 0.0);
    if (sample.second >= 0 && sample.second < OUTPUT_SIZE) {
      y_onehot[sample.second] = 1.0;
    }
    Y.push_back(y_onehot);
  }

  // Normalize training data
  normalizeTrainingData(X);

  size_t n         = X.size();
  double prev_loss = 999999.0;

  double initial_lr       = learning_rate;
  double lr_decay         = 0.95;
  int    patience         = 50;
  int    no_improve_count = 0;
  int    loss_counting    = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937       gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t idx = 0; idx < n; ++idx) {
      size_t      i = indices[idx];
      const auto& x = X[i];
      const auto& y = Y[i];

      // Forward pass through all layers
      // Hidden layer 1
      std::vector<double> z1(HIDDEN1_SIZE), a1(HIDDEN1_SIZE);
      for (int h = 0; h < HIDDEN1_SIZE; ++h) {
        z1[h] = dot(W1[h], x) + b1[h];
        a1[h] = sigmoid(z1[h]);
      }

      // Hidden layer 2
      std::vector<double> z2(HIDDEN2_SIZE), a2(HIDDEN2_SIZE);
      for (int h = 0; h < HIDDEN2_SIZE; ++h) {
        z2[h] = dot(W2[h], a1) + b2[h];
        a2[h] = sigmoid(z2[h]);
      }

      // Hidden layer 3
      std::vector<double> z3(HIDDEN3_SIZE), a3(HIDDEN3_SIZE);
      for (int h = 0; h < HIDDEN3_SIZE; ++h) {
        z3[h] = dot(W3[h], a2) + b3[h];
        a3[h] = sigmoid(z3[h]);
      }

      // Output layer
      std::vector<double> z4(OUTPUT_SIZE);
      for (int o = 0; o < OUTPUT_SIZE; ++o) {
        z4[o] = dot(W4[o], a3) + b4[o];
      }

      std::vector<double> y_hat = softmax(z4);

      // Calculate cross-entropy loss
      double sample_loss = 0.0;
      for (int o = 0; o < OUTPUT_SIZE; ++o) {
        if (y[o] > 0) {
          sample_loss += -y[o] * log(y_hat[o] + 1e-15);
        }
      }
      total_loss += sample_loss;

      // Backpropagation
      // Output layer error
      std::vector<double> error4(OUTPUT_SIZE);
      for (int o = 0; o < OUTPUT_SIZE; ++o) {
        error4[o] = y_hat[o] - y[o];
      }

      // Output layer gradients
      std::vector<std::vector<double>> dW4(OUTPUT_SIZE, std::vector<double>(HIDDEN3_SIZE));
      std::vector<double>              db4(OUTPUT_SIZE);

      for (int o = 0; o < OUTPUT_SIZE; ++o) {
        db4[o] = error4[o];
        for (int h = 0; h < HIDDEN3_SIZE; ++h) {
          dW4[o][h] = error4[o] * a3[h];
        }
      }

      // Hidden layer 3 error
      std::vector<double> error3(HIDDEN3_SIZE, 0.0);
      for (int h = 0; h < HIDDEN3_SIZE; ++h) {
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
          error3[h] += W4[o][h] * error4[o];
        }
        error3[h] *= sigmoid_derivative(z3[h]);
      }

      // Hidden layer 3 gradients
      std::vector<std::vector<double>> dW3(HIDDEN3_SIZE, std::vector<double>(HIDDEN2_SIZE));
      std::vector<double>              db3(HIDDEN3_SIZE);

      for (int h = 0; h < HIDDEN3_SIZE; ++h) {
        db3[h] = error3[h];
        for (int j = 0; j < HIDDEN2_SIZE; ++j) {
          dW3[h][j] = error3[h] * a2[j];
        }
      }

      // Hidden layer 2 error
      std::vector<double> error2(HIDDEN2_SIZE, 0.0);
      for (int h = 0; h < HIDDEN2_SIZE; ++h) {
        for (int k = 0; k < HIDDEN3_SIZE; ++k) {
          error2[h] += W3[k][h] * error3[k];
        }
        error2[h] *= sigmoid_derivative(z2[h]);
      }

      // Hidden layer 2 gradients
      std::vector<std::vector<double>> dW2(HIDDEN2_SIZE, std::vector<double>(HIDDEN1_SIZE));
      std::vector<double>              db2(HIDDEN2_SIZE);

      for (int h = 0; h < HIDDEN2_SIZE; ++h) {
        db2[h] = error2[h];
        for (int j = 0; j < HIDDEN1_SIZE; ++j) {
          dW2[h][j] = error2[h] * a1[j];
        }
      }

      // Hidden layer 1 error
      std::vector<double> error1(HIDDEN1_SIZE, 0.0);
      for (int h = 0; h < HIDDEN1_SIZE; ++h) {
        for (int k = 0; k < HIDDEN2_SIZE; ++k) {
          error1[h] += W2[k][h] * error2[k];
        }
        error1[h] *= sigmoid_derivative(z1[h]);
      }

      // Hidden layer 1 gradients
      std::vector<std::vector<double>> dW1(HIDDEN1_SIZE, std::vector<double>(INPUT_SIZE));
      std::vector<double>              db1(HIDDEN1_SIZE);

      for (int h = 0; h < HIDDEN1_SIZE; ++h) {
        db1[h] = error1[h];
        for (int j = 0; j < INPUT_SIZE; ++j) {
          dW1[h][j] = error1[h] * x[j];
        }
      }

      // Update all parameters
      add_to(W4, dW4, -learning_rate);
      add_to(b4, db4, -learning_rate);
      add_to(W3, dW3, -learning_rate);
      add_to(b3, db3, -learning_rate);
      add_to(W2, dW2, -learning_rate);
      add_to(b2, db2, -learning_rate);
      add_to(W1, dW1, -learning_rate);
      add_to(b1, db1, -learning_rate);
      
      loss_counting++;
      if (loss_counting % 10000 == 0) {
        loss_history.push_back(total_loss);
      }
    }

    // Calculate accuracy
    int correct = 0;
    for (size_t i = 0; i < n; ++i) {
      int pred   = predictClass(X[i]);
      int actual = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
      if (pred == actual)
        correct++;
    }

    double avg_loss = total_loss / n;
    double accuracy = 100.0 * correct / n;

    // Store loss and accuracy
    // loss_history.push_back(avg_loss);
    accuracy_history.push_back(accuracy);

    // Adaptive learning rate
    if (avg_loss >= prev_loss) {
      no_improve_count++;
      if (no_improve_count >= patience) {
        learning_rate *= lr_decay;
        no_improve_count = 0;
        std::cout << "Reducing learning rate to: " << learning_rate << std::endl;
      }
    } else {
      no_improve_count = 0;
    }

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      double loss_change = prev_loss - avg_loss;
      std::cout << "Epoch " << epoch + 1 << ": Loss = " << avg_loss << ", Accuracy = " << accuracy << "%, LR = " << learning_rate << std::endl;

      if (progress_callback) {
        progress_callback(epoch + 1, avg_loss, learning_rate, loss_change);
      }

      prev_loss = avg_loss;
    }

    // Early stopping if we achieve high accuracy
    if (avg_loss < 1e-4) {
      std::cout << "Early stopping - 1e-6 achieved!" << std::endl;
      break;
    }
  }

  // Reset learning rate
  learning_rate = initial_lr;
}