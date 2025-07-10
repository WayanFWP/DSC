#ifndef MLP_EBPA_HPP
#define MLP_EBPA_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "dataset.hpp"

using namespace std;
namespace fs = filesystem;

struct Prediction {
  int            predicted_label;
  double         confidence;
  double         error;
  double         max_probability;
  vector<double> all_probabilities;
};

struct AllResults {
  // Training errors
  vector<double> training_errors_per_iteration;
  vector<int>    iteration_numbers;  // Add this line
  vector<double> training_errors_per_epoch;

  // Test results
  vector<string>         test_filenames;
  vector<char>           predicted_letters;
  vector<char>           actual_letters;
  vector<double>         detection_errors;
  vector<double>         confidences;
  vector<double>         max_probabilities;
  vector<vector<double>> all_class_probabilities;

  // User input results
  vector<double> user_detection_errors;
  vector<double> user_confidences;
  vector<char>   user_predictions;
};

AllResults global_results;

class MLP {
 private:
  int                    input_size, hidden_size, output_size;
  vector<vector<double>> weights_ih, weights_ho;
  vector<double>         bias_h, bias_o;
  vector<double>         hidden, output;
  double                 learning_rate;

  // Add these to store training errors
  vector<double> epoch_errors;
  vector<double> iteration_errors;    // Add this line
  vector<int>    iteration_numbers;   // Add this line

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
    random_device rd;
    mt19937       gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);

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

  void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
    epoch_errors.clear();
    iteration_errors.clear();
    iteration_numbers.clear();

    int iteration_count = 0;

    for (int e = 0; e < epochs; ++e) {
      double total_error = 0;

      for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass
        forward(inputs[i]);

        // Calculate error for this iteration
        double iteration_error = 0;
        for (int j = 0; j < output_size; ++j) {
          double error = targets[i][j] - output[j];
          iteration_error += error * error;  // MSE for this sample
        }

        // Store iteration error
        iteration_errors.push_back(iteration_error);
        iteration_numbers.push_back(iteration_count++);
        total_error += iteration_error;

        // Backward pass
        backward(inputs[i], targets[i]);
      }

      double avg_error = total_error / inputs.size();
      epoch_errors.push_back(avg_error);

      // Print progress every 50 epochs
      if ((e + 1) % 50 == 0 || e == epochs - 1) {
        cout << "Epoch " << e + 1 << " - Avg Error: " << avg_error << " - Total Iterations: " << iteration_count << endl;
      }
    }

    cout << "Total iterations completed: " << iteration_count << endl;
  }

  // Get training errors for plotting - these should be member functions
  vector<double> get_training_errors() const { 
    return epoch_errors; 
  }
  
  vector<double> get_iteration_errors() const {
    return iteration_errors;
  }

  vector<int> get_iteration_numbers() const {
    return iteration_numbers;
  }

  int predict(const vector<double>& input) {
    vector<double> result = forward(input);
    return distance(result.begin(), max_element(result.begin(), result.end()));
  }

  // Function to validate the prediction and return detailed results
  Prediction validation(const vector<double>& input) {
    vector<double> result = forward(input);
    Prediction     prediction;

    auto max_it                  = max_element(result.begin(), result.end());
    prediction.predicted_label   = distance(result.begin(), max_it);
    prediction.max_probability   = *max_it;
    prediction.all_probabilities = result;

    prediction.confidence = prediction.max_probability * 100;  // Convert to percentage
    prediction.error      = 1 - prediction.max_probability;    // Error is the complement of the max probability
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

  // Clear previous test results
  global_results.test_filenames.clear();
  global_results.predicted_letters.clear();
  global_results.actual_letters.clear();
  global_results.detection_errors.clear();
  global_results.confidences.clear();
  global_results.max_probabilities.clear();
  global_results.all_class_probabilities.clear();

  for (const auto& entry : fs::directory_iterator("testcase")) {
    string test_filename = entry.path().filename().string();
    cout << "📂 Testing file: " << test_filename << endl;

    vector<vector<double>> test_matrix       = load_test_matrix(test_filename);
    auto                   prediction_result = mlp.validation(flatten(test_matrix));

    // Store ALL results
    global_results.test_filenames.push_back(test_filename);
    global_results.predicted_letters.push_back(char('A' + prediction_result.predicted_label));
    global_results.detection_errors.push_back(prediction_result.error);
    global_results.confidences.push_back(prediction_result.confidence);
    global_results.max_probabilities.push_back(prediction_result.max_probability);
    global_results.all_class_probabilities.push_back(prediction_result.all_probabilities);

    // Extract actual letter from filename if possible
    if (test_filename.length() > 0 && test_filename[0] >= 'A' && test_filename[0] <= 'H') {
      global_results.actual_letters.push_back(test_filename[0]);
    }

    cout << "✅ Predicted Letter: " << char('A' + prediction_result.predicted_label) << endl;
    cout << "📊 Confidence: " << fixed << setprecision(2) << prediction_result.confidence << "%" << endl;
    cout << "⚠️  Detection Error: " << fixed << setprecision(4) << prediction_result.error << endl;
    cout << "🎯 Max Probability: " << fixed << setprecision(4) << prediction_result.max_probability << endl;

    cout << "📈 All Class Probabilities:" << endl;
    for (int i = 0; i < prediction_result.all_probabilities.size(); ++i) {
      cout << "   " << char('A' + i) << ": " << fixed << setprecision(4) << prediction_result.all_probabilities[i] << endl;
    }
    cout << endl;
  }
}

// Function to predict user-inputted matrix
void predict_user_input(MLP& mlp) {
  cout << "Enter your own 12x12 matrix to predict: " << endl;
  vector<vector<double>> user_matrix = get_user_matrix();

  auto prediction_result = mlp.validation(flatten(user_matrix));

  // Store user input results
  global_results.user_detection_errors.push_back(prediction_result.error);
  global_results.user_confidences.push_back(prediction_result.confidence);
  global_results.user_predictions.push_back(char('A' + prediction_result.predicted_label));

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
    cout << "   " << char('A' + i) << ": " << fixed << setprecision(4) << prediction_result.all_probabilities[i] << endl;
  }
}

void save_all_results() {
  // Save training errors per iteration
  ofstream iteration_file("training_errors_per_iteration.csv");
  if (iteration_file) {
    iteration_file << "Iteration,Training_Error\n";
    for (size_t i = 0; i < global_results.training_errors_per_iteration.size(); ++i) {
      iteration_file << global_results.iteration_numbers[i] + 1 << "," 
                    << fixed << setprecision(8) 
                    << global_results.training_errors_per_iteration[i] << "\n";
    }
    iteration_file.close();
    cout << "📁 Training errors per iteration saved to training_errors_per_iteration.csv" << endl;
  }

  // Save training errors per epoch
  ofstream training_file("training_errors_per_epoch.csv");
  if (training_file) {
    training_file << "Epoch,Training_Error\n";
    for (size_t i = 0; i < global_results.training_errors_per_epoch.size(); ++i) {
      training_file << i + 1 << "," << fixed << setprecision(8) 
                   << global_results.training_errors_per_epoch[i] << "\n";
    }
    training_file.close();
    cout << "📁 Training errors per epoch saved to training_errors_per_epoch.csv" << endl;
  }

  // Save test results
  ofstream test_file("test_results.csv");
  if (test_file) {
    test_file << "Filename,Predicted_Letter,Actual_Letter,Detection_Error,Confidence,Max_Probability";
    for (int i = 0; i < 8; ++i) {
      test_file << ",Prob_" << char('A' + i);
    }
    test_file << "\n";
    
    for (size_t i = 0; i < global_results.test_filenames.size(); ++i) {
      test_file << global_results.test_filenames[i] << ","
               << global_results.predicted_letters[i] << ",";
      
      if (i < global_results.actual_letters.size()) {
        test_file << global_results.actual_letters[i];
      }
      test_file << "," << fixed << setprecision(8) << global_results.detection_errors[i]
               << "," << fixed << setprecision(2) << global_results.confidences[i]
               << "," << fixed << setprecision(8) << global_results.max_probabilities[i];
               
      // Add all class probabilities
      if (i < global_results.all_class_probabilities.size()) {
        for (double prob : global_results.all_class_probabilities[i]) {
          test_file << "," << fixed << setprecision(8) << prob;
        }
      }
      test_file << "\n";
    }
    test_file.close();
    cout << "📁 Test results saved to test_results.csv" << endl;
  }

  // Save user input results
  if (!global_results.user_detection_errors.empty()) {
    ofstream user_file("user_input_results.csv");
    if (user_file) {
      user_file << "Input_Number,Predicted_Letter,Detection_Error,Confidence\n";
      for (size_t i = 0; i < global_results.user_detection_errors.size(); ++i) {
        user_file << i + 1 << "," << global_results.user_predictions[i] 
                 << "," << fixed << setprecision(8) << global_results.user_detection_errors[i]
                 << "," << fixed << setprecision(2) << global_results.user_confidences[i] << "\n";
      }
      user_file.close();
      cout << "📁 User input results saved to user_input_results.csv" << endl;
    }
  }
}

void print_all_statistics() {
  cout << "\n📊 COMPREHENSIVE ERROR ANALYSIS:" << endl;

  // Training error statistics per iteration
  if (!global_results.training_errors_per_iteration.empty()) {
    double min_iter   = *min_element(global_results.training_errors_per_iteration.begin(), 
                                   global_results.training_errors_per_iteration.end());
    double max_iter   = *max_element(global_results.training_errors_per_iteration.begin(), 
                                   global_results.training_errors_per_iteration.end());
    double final_iter = global_results.training_errors_per_iteration.back();

    cout << "\n🔄 Training Errors (Per Iteration):" << endl;
    cout << "   📈 Initial Error: " << fixed << setprecision(6) 
         << global_results.training_errors_per_iteration[0] << endl;
    cout << "   📉 Final Error: " << fixed << setprecision(6) << final_iter << endl;
    cout << "   📊 Min Error: " << fixed << setprecision(6) << min_iter << endl;
    cout << "   📊 Max Error: " << fixed << setprecision(6) << max_iter << endl;
    cout << "   📋 Total Iterations: " << global_results.training_errors_per_iteration.size() << endl;
  }

  // Training error statistics per epoch
  if (!global_results.training_errors_per_epoch.empty()) {
    double min_train   = *min_element(global_results.training_errors_per_epoch.begin(), 
                                    global_results.training_errors_per_epoch.end());
    double max_train   = *max_element(global_results.training_errors_per_epoch.begin(), 
                                    global_results.training_errors_per_epoch.end());
    double final_train = global_results.training_errors_per_epoch.back();

    cout << "\n🎯 Training Errors (Per Epoch):" << endl;
    cout << "   📈 Initial Error: " << fixed << setprecision(6) 
         << global_results.training_errors_per_epoch[0] << endl;
    cout << "   📉 Final Error: " << fixed << setprecision(6) << final_train << endl;
    cout << "   📊 Min Error: " << fixed << setprecision(6) << min_train << endl;
    cout << "   📊 Max Error: " << fixed << setprecision(6) << max_train << endl;
    cout << "   📋 Total Epochs: " << global_results.training_errors_per_epoch.size() << endl;
  }

  // Test error statistics
  if (!global_results.detection_errors.empty()) {
    double sum = 0, min_error = global_results.detection_errors[0], max_error = global_results.detection_errors[0];
    for (double error : global_results.detection_errors) {
      sum += error;
      min_error = min(min_error, error);
      max_error = max(max_error, error);
    }
    double mean_error = sum / global_results.detection_errors.size();

    cout << "\n🔍 Test Detection Errors:" << endl;
    cout << "   📈 Mean Error: " << fixed << setprecision(6) << mean_error << endl;
    cout << "   📉 Min Error: " << fixed << setprecision(6) << min_error << endl;
    cout << "   📈 Max Error: " << fixed << setprecision(6) << max_error << endl;
    cout << "   📋 Total Tests: " << global_results.detection_errors.size() << endl;
  }

  // User input error statistics
  if (!global_results.user_detection_errors.empty()) {
    double sum = 0, min_error = global_results.user_detection_errors[0], max_error = global_results.user_detection_errors[0];
    for (double error : global_results.user_detection_errors) {
      sum += error;
      min_error = min(min_error, error);
      max_error = max(max_error, error);
    }
    double mean_error = sum / global_results.user_detection_errors.size();

    cout << "\n👤 User Input Errors:" << endl;
    cout << "   📈 Mean Error: " << fixed << setprecision(6) << mean_error << endl;
    cout << "   📉 Min Error: " << fixed << setprecision(6) << min_error << endl;
    cout << "   📈 Max Error: " << fixed << setprecision(6) << max_error << endl;
    cout << "   📋 Total Inputs: " << global_results.user_detection_errors.size() << endl;
  }
}

#endif  // MLP_EBPA_HPP