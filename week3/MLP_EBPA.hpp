#ifndef MLP_EBPA_HPP
#define MLP_EBPA_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "dataset.hpp"

using namespace std;
namespace fs = filesystem;

/**
 * @class MLP
 * @brief Implements a Multi-layer Perceptron (MLP) with one hidden layer.
 *
 * This class provides a simple neural network with:
 * - Randomized initialization of weights and biases in the range [-1.0, 1.0].
 * - Forward propagation using the sigmoid activation function.
 * - Backward propagation (backpropagation) using the derivative of the sigmoid function to update weights and biases.
 * - Training over multiple epochs with reporting of the average error per epoch.
 * - Prediction by selecting the output neuron with the highest activation.
 */

/**
 * @brief Constructs an MLP with the specified number of input, hidden, and output neurons.
 *
 * During construction, the network's weights and biases are initialized to random values from a uniform distribution ranging from -1.0 to 1.0.
 *
 * @param input  The number of neurons in the input layer.
 * @param hidden The number of neurons in the hidden layer.
 * @param output The number of neurons in the output layer.
 * @param lr     The learning rate for training (default value is 0.1).
 */
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

  /**
   * @brief Computes the forward propagation of the neural network.
   *
   * This function performs a forward pass through the neural network by first computing the
   * activations of the hidden layer using the input vector, weights, and biases, and then
   * applying the sigmoid activation function. It proceeds to compute the output layer in a
   * similar fashion by using the hidden layer activations, corresponding weights, and biases,
   * followed by the sigmoid activation.
   *
   * @param input A constant reference to a vector of doubles representing the input features.
   * @return A vector of doubles representing the activated outputs from the network.
   */
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

  /**
   * @brief Performs backward propagation to update the neural network's weights and biases.
   *
   * This function calculates the error and corresponding deltas for both the output and 
   * hidden layers by comparing the target outputs with the actual outputs computed 
   * during the forward pass. It applies the derivative of the sigmoid function 
   * to adjust the deltas and then updates the weights connecting the hidden and 
   * output layers as well as the weights connecting the input and hidden layers. 
   * Additionally, it updates the bias values for the hidden and output layers.
   *
   * @param input  The input vector that was provided during the network's forward pass.
   * @param target The expected output vector used to compute the output error.
   */
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

  /**
   * @brief Trains the neural network using the provided inputs and targets over a given number of epochs.
   *
   * For each epoch, this function iterates through all training examples, performs a forward pass,
   * computes the error with a backward pass, and aggregates the total error across all outputs.
   * The average error for each epoch is then printed to the console.
   *
   * @param inputs A vector of vectors containing the input data for each training example.
   * @param targets A vector of vectors containing the expected output data for each training example.
   * @param epochs The number of times to iterate over the entire training dataset.
   */
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

  /**
   * Computes the predicted class index from the provided input features.
   *
   * This function performs a forward pass through the network using the given input,
   * then finds the index of the maximum element in the resulting output vector to 
   * determine the predicted class.
   *
   * @param input A constant reference to a vector of doubles that represents 
   *              the input features for the prediction.
   * @return An integer indicating the index of the predicted class.
   */
  int predict(const vector<double>& input) {
    vector<double> result = forward(input);
    return distance(result.begin(), max_element(result.begin(), result.end()));
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
  for (const auto& entry : fs::directory_iterator("test")) {
    string test_filename = entry.path().filename().string();
    cout << "📂 Testing file: " << test_filename << endl;

    vector<vector<double>> test_matrix     = load_test_matrix(test_filename);
    int                    predicted_label = mlp.predict(flatten(test_matrix));

    cout << "✅ Predicted Letter: " << char('A' + predicted_label) << "\n" << endl;
  }
}

// Function to predict user-inputted matrix
void predict_user_input(MLP& mlp) {
  cout << "Enter your own 12x12 matrix to predict: " << endl;
  vector<vector<double>> user_matrix = get_user_matrix();

  int predicted_label = mlp.predict(flatten(user_matrix));
  cout << "🎯 Predicted Letter: " << char('A' + predicted_label) << endl;
}

#endif  // MLP_EBPA_HPP
