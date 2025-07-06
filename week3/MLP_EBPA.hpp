#ifndef MLP_EBPA_HPP
#define MLP_EBPA_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#include "dataset.hpp"

using namespace std;
namespace fs = filesystem;

struct Prediction{
  int predicted_label;
  double confidence;
  double error;
  double max_probability;
  vector<double> all_probabilities;
};

class MLP {
 private:
  int                    input_size, hidden_size, output_size;
  vector<vector<double>> weights_ih, weights_ho;
  vector<double>         bias_h, bias_o;
  vector<double>         hidden, output;
  double                 learning_rate;

  double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
  double sigmoid_derivative(double x) { return x * (1.0 - x); }

 public:
  MLP(int input, int hidden, int output, double lr = 0.1)
      : input_size(input),
        hidden_size(hidden),
        output_size(output),
        learning_rate(lr),
        hidden(hidden_size, 0.0),
        output(output_size, 0.0),
        bias_h(hidden_size, 0.0),
        bias_o(output_size, 0.0),
        weights_ih(input, vector<double>(hidden)),
        weights_ho(hidden, vector<double>(output)) {
    random_device rd;  // Creates a non-deterministic random device which is typically used to produce a seed.
    mt19937       gen(
        rd());  // Initializes an instance of the Mersenne Twister 19937 generator with the seed from rd. Mersenne Twister is a widely used pseudorandom number generator.
    uniform_real_distribution<double>
        dist(-1.0, 1.0);  // Defines a uniform real distribution that will produce random double values ranging from -1.0 to 1.0 each time it is used.

    for (int i = 0; i < input_size; ++i)
      for (int j = 0; j < hidden_size; ++j) weights_ih[i][j] = dist(gen);

    for (int i = 0; i < hidden_size; ++i) {
      for (int j = 0; j < output_size; ++j) weights_ho[i][j] = dist(gen);
      bias_h[i] = dist(gen);
    }

    for (int i = 0; i < output_size; ++i) bias_o[i] = dist(gen);
  }
  
  // Forward pass to compute output
  vector<double> forward(const vector<double>& input) {
    for (int i = 0; i < hidden_size; ++i) {
      hidden[i] = bias_h[i];
      for (int j = 0; j < input_size; ++j) hidden[i] += input[j] * weights_ih[j][i];
      hidden[i] = sigmoid(hidden[i]);
    }

    for (int i = 0; i < output_size; ++i) {
      output[i] = bias_o[i];
      for (int j = 0; j < hidden_size; ++j) output[i] += hidden[j] * weights_ho[j][i];
      output[i] = sigmoid(output[i]);
    }
    return output;
  }
  // Backward pass to update weights and biases
  void backward(const vector<double>& input, const vector<double>& target) {
    vector<double> output_error(output_size), output_delta(output_size);
    vector<double> hidden_error(hidden_size), hidden_delta(hidden_size);

    for (int i = 0; i < output_size; ++i) {
      output_error[i] = target[i] - output[i];
      output_delta[i] = output_error[i] * sigmoid_derivative(output[i]);
    }

    for (int i = 0; i < hidden_size; ++i) {
      hidden_error[i] = 0;
      for (int j = 0; j < output_size; ++j) hidden_error[i] += output_delta[j] * weights_ho[i][j];
      hidden_delta[i] = hidden_error[i] * sigmoid_derivative(hidden[i]);
    }

    for (int i = 0; i < hidden_size; ++i)
      for (int j = 0; j < output_size; ++j) weights_ho[i][j] += learning_rate * output_delta[j] * hidden[i];

    for (int i = 0; i < input_size; ++i)
      for (int j = 0; j < hidden_size; ++j) weights_ih[i][j] += learning_rate * hidden_delta[j] * input[i];

    for (int i = 0; i < output_size; ++i) bias_o[i] += learning_rate * output_delta[i];

    for (int i = 0; i < hidden_size; ++i) bias_h[i] += learning_rate * hidden_delta[i];
  }

  // Function to train the MLP model
  void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
    for (int e = 0; e < epochs; ++e) {
      double total_error = 0;
      for (size_t i = 0; i < inputs.size(); ++i) {
        forward(inputs[i]);
        backward(inputs[i], targets[i]);
        for (int j = 0; j < output_size; ++j) total_error += pow(targets[i][j] - output[j], 2);
      }
      if (e == epochs - 1)
        cout << "Epoch " << e + 1 << " - Error: " << total_error / inputs.size() << endl;
    }
  }

  int predict(const vector<double>& input) {
    vector<double> result = forward(input);
    return distance(result.begin(), max_element(result.begin(), result.end()));
  }
  // Function to validate the prediction and return detailed results
  Prediction validation(const vector<double>& input){
    vector<double> result = forward(input);
    Prediction prediction;

    auto max_it = max_element(result.begin(), result.end());
    prediction.predicted_label = distance(result.begin(), max_it);
    prediction.max_probability = *max_it;
    prediction.all_probabilities = result;

    prediction.confidence = prediction.max_probability * 100; // Convert to percentage
    prediction.error = 1 - prediction.max_probability; // Error is the complement of the max probability
    return prediction;
  }
};

// Function to load a 12x12 matrix from user input
vector<vector<double>> get_user_matrix() {
  vector<vector<double>> user_matrix(12, vector<double>(12, 0));

  cout << "Enter a 12x12 matrix (only 0s and 1s):" << endl;
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 12; ++j) {
      cin >> user_matrix[i][j];
      if (user_matrix[i][j] != 0 && user_matrix[i][j] != 1) {
        cout << "Invalid input! Please enter only 0 or 1." << endl;
        return get_user_matrix();  // Retry input if invalid
      }
    }
  }

  return user_matrix;
}

// Function to display training dataset as ASCII visualization
void display_dataset(const vector<vector<vector<double>>>& letters) {
  cout << "📌 Input Data: \n";
  for (size_t i = 0; i < letters.size(); ++i) {
    cout << "Letter: " << char('A' + i) << endl;
    for (const auto& row : letters[i]) {
      for (double pixel : row) {
        cout << (pixel == 1 ? '*' : ' ');
      }
      cout << endl;
    }
    cout << endl;
  }
}

// Function to test all files in "testcase/" directory
void test_files(MLP& mlp) {
  cout << "\n🔍 Loading test matrices from 'testcase/' folder...\n";
  for (const auto& entry : fs::directory_iterator("testcase")) {
    string test_filename = entry.path().filename().string();
    cout << "📂 Testing file: " << test_filename << endl;

    vector<vector<double>> test_matrix = load_test_matrix(test_filename);
    auto prediction_result = mlp.validation(flatten(test_matrix));

    cout << "✅ Predicted Letter: " << char('A' + prediction_result.predicted_label) << endl;
    cout << "📊 Confidence: " << fixed << setprecision(2) << prediction_result.confidence << "%" << endl;
    cout << "⚠️  Detection Error: " << fixed << setprecision(4) << prediction_result.error << endl;
    cout << "🎯 Max Probability: " << fixed << setprecision(4) << prediction_result.max_probability << endl;
    
    // Show all class probabilities
    cout << "📈 All Class Probabilities:" << endl;
    for (int i = 0; i < prediction_result.all_probabilities.size(); ++i) {
      cout << "   " << char('A' + i) << ": " << fixed << setprecision(4) 
           << prediction_result.all_probabilities[i] << endl;
    }
    cout << endl;
  }
}

// Function to predict user-inputted matrix
void predict_user_input(MLP& mlp) {
  cout << "Enter your own 12x12 matrix to predict: " << endl;
  vector<vector<double>> user_matrix = get_user_matrix();

  auto prediction_result = mlp.validation(flatten(user_matrix));
  
  cout << "🎯 Predicted Letter: " << char('A' + prediction_result.predicted_label) << endl;
  cout << "📊 Confidence: " << fixed << setprecision(2) << prediction_result.confidence << "%" << endl;
  cout << "⚠️  Detection Error: " << fixed << setprecision(4) << prediction_result.error << endl;
  cout << "🎯 Max Probability: " << fixed << setprecision(4) << prediction_result.max_probability << endl;
  
  // Show confidence level interpretation
  if (prediction_result.confidence >= 90) {
    cout << "🟢 High Confidence - Very reliable prediction!" << endl;
  } else if (prediction_result.confidence >= 70) {
    cout << "🟡 Medium Confidence - Fairly reliable prediction." << endl;
  } else if (prediction_result.confidence >= 50) {
    cout << "🟠 Low Confidence - Prediction may be uncertain." << endl;
  } else {
    cout << "🔴 Very Low Confidence - Prediction is highly uncertain!" << endl;
  }
  
  cout << "📈 All Class Probabilities:" << endl;
  for (int i = 0; i < prediction_result.all_probabilities.size(); ++i) {
    cout << "   " << char('A' + i) << ": " << fixed << setprecision(4) 
         << prediction_result.all_probabilities[i] << endl;
  }
}

#endif  // MLP_EBPA_HPP
