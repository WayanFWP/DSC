#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

class NeuralNetwork {
 private:
  int                    num_inputs;
  int                    num_perceptrons;
  vector<vector<double>> weight_matrix;
  double                 real_output = 0.0;

  void initialize_weights() {
    for (int i = 0; i < num_inputs; i++) {
      for (int j = 0; j < num_perceptrons; j++) {
        int bit_value       = (j >> i) & 1;  // Extract bit i dari number j
        weight_matrix[i][j] = bit_value ? 1 : -1;
      }
    }
    cout << "initialized " << num_perceptrons << " perceptrons with balanced weights." << endl;
  }

 public:
  NeuralNetwork(int inputs, int perceptrons) : num_inputs(inputs), num_perceptrons(perceptrons) {
    weight_matrix.resize(num_inputs, vector<double>(num_perceptrons, 0.0));
    initialize_weights();
  }

  vector<int> compute_hidden_layer_sum(const vector<int> &input_user) {
    vector<int> hidden_layer_sum(num_perceptrons, 0);
    for (int j = 0; j < num_perceptrons; j++) {
      for (int i = 0; i < num_inputs; i++) {
        hidden_layer_sum[j] += input_user[i] * weight_matrix[i][j];
      }
    }
    return hidden_layer_sum;
  }

  vector<int> apply_thresholding(const vector<int> &hidden_layer_sum, int theta) {
    vector<int> hidden_layer_output(num_perceptrons, 0);
    for (int j = 0; j < num_perceptrons; j++) {
      hidden_layer_output[j] = (hidden_layer_sum[j] >= theta) ? 1 : 0;
    }
    return hidden_layer_output;
  }

  double compute_final_output(const vector<int> &hidden_layer_output) {
    int output_layer_sum = 0;  // Reset output layer sum for each computation
    for (int i = 0; i < num_perceptrons; i++) {
      output_layer_sum += hidden_layer_output[i];
    }
    cout << "Active perceptrons: " << output_layer_sum << endl;
    // double confident = static_cast<double>(output_layer_sum) / num_perceptrons;

    real_output = static_cast<double>(output_layer_sum) / num_perceptrons;  // Update real_output

    int threshold_output = num_perceptrons / 2;
    return (output_layer_sum >= threshold_output) ? 1 : 0;
  }

  double get_output() { return real_output; }

  void testcase() {
    srand(time(0));
    vector<int> input_user;
    int         value = 0;
    for (int i = 0; i < num_inputs; i++) {
      input_user.push_back(rand() % 2);
      cout << input_user[i] << " ";
      if (input_user[i] == 1)
        value += 1;
    }
    cout << "\ninput 1 is " << value << endl;
    vector<int> hidden_layer_sum    = compute_hidden_layer_sum(input_user);
    vector<int> hidden_layer_output = apply_thresholding(hidden_layer_sum, 1);
    int         final_output        = compute_final_output(hidden_layer_output);
    double      output              = get_output();
    cout << "real output is " << output << endl;
    cout << "output is " << final_output << endl;
  }

  void inputUser() {
    vector<int> inputs;
    int         value;
    cout << "Enter " << num_inputs << " input Binary (0/1): " << endl;
    int count = 0;

    for (int i = 0; i < num_inputs; i++) {
      cin >> value;
      inputs.push_back(value % 2);
      if (value == 1)
        count += 1;
    }
    cout << "input 1 is " << count << endl;
    vector<int> hidden_layer_sum    = compute_hidden_layer_sum(inputs);
    vector<int> hidden_layer_output = apply_thresholding(hidden_layer_sum, 1);
    int         final_output        = compute_final_output(hidden_layer_output);
    double      output              = get_output();
    cout << "real output is " << output << endl;

    cout << "output is " << final_output << endl;
  }

  void testAllPossibilities(NeuralNetwork &nn) {
    ofstream file("output.csv");  // Open file for writing
    if (!file.is_open()) {
      cerr << "Failed to open file for writing!" << endl;
      return;
    }

    // Write CSV header - ADD REAL_OUTPUT COLUMN
    file << "Input1,Input2,Input3,Input4,Input5,Input6,Input7,Input8,Input9,Input10,Input11,Output,Real_Output\n";

    for (int i = 0; i < (1 << 11); i++) {  // 2048 inputs
      vector<int> input_user;
      for (int bit = 10; bit >= 0; bit--) {
        input_user.push_back((i >> bit) & 1);
      }

      vector<int> hidden_sum  = nn.compute_hidden_layer_sum(input_user);
      vector<int> thresholded = nn.apply_thresholding(hidden_sum, 1);
      int         output      = nn.compute_final_output(thresholded);
      double      real_output = nn.get_output();

      // Write to file
      for (int b = 0; b < input_user.size(); b++) {
        file << input_user[b];
        if (b < input_user.size() - 1)
          file << ",";
      }
      file << "," << output << "," << real_output << "\n";
    }

    file.close();  // Close the file
    cout << "Output written to output.csv" << endl;
  }
};

int main() {
  int           INPUT_SIZE      = 11;
  int           PERCEPTRON_SIZE = 2048;
  NeuralNetwork nn(INPUT_SIZE, PERCEPTRON_SIZE);
  cout << endl;

  char choice;

  cout << "Do you want to input the binary input? (y/n): ";
  cin >> choice;
  if (choice == 'y' || choice == 'Y')
    nn.inputUser();
  else if (choice == 't' || choice == 'T') {
    cout << "Testing all possibilities..." << endl;
    nn.testAllPossibilities(nn);
  } else
    nn.testcase();

  return 0;
}