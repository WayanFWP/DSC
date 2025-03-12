#include <iostream>
#include <vector>

using namespace std;

class PerceptronModel {
private:
  int num_inputs;
  int num_perceptrons;
  vector<vector<int>> weight_matrix;
  vector<int> input_user;
  vector<int> hidden_layer_sum;
  vector<int> hidden_layer_output;

  // Initialize the weight matrix with alternating +1 and -1 pattern
  void initializeWeights() {
    weight_matrix.resize(num_inputs, vector<int>(num_perceptrons, 0));
    for (int i = 0; i < num_inputs; i++) {
      int block_size = 1 << i;  // 2^i
      int weight_value = -1;
      for (int j = 0; j < num_perceptrons; j++) {
        weight_matrix[i][j] = weight_value;
        if ((j + 1) % block_size == 0) {
          weight_value *= -1;
        }
      }
    }
  }

  vector<int> testcase() {
    srand(time(0));
    input_user.resize(num_inputs, 0);
    int value = 0;
    for (int i = 0; i < num_inputs; i++) {
      input_user[i] = rand() % 2;
      cout << input_user[i] << " ";
      if (input_user[i] == 1)
        value++;
    }
    cout << endl;
    cout << "input 1 is " << value << endl;
    return input_user;
  }

  // Get binary input from the user with validation
  vector<int> getBinaryInputs() {
    vector<int> inputs;
    int value;
    cout << "Masukkan " << num_inputs << " digit biner (0/1): " << endl;
    for (int i = 0; i < num_inputs; i++) {
      while (true) {
        cout << "Input " << i + 1 << ": ";
        cin >> value;
        if (value == 0 || value == 1) {
          inputs.push_back(value);
          break;
        } else {
          cout << "Error: Input harus 0 atau 1!" << endl;
        }
      }
    }
    return inputs;
  }

public:
  PerceptronModel(int inputs, int perceptrons)
    : num_inputs(inputs), num_perceptrons(perceptrons),
      hidden_layer_sum(perceptrons, 0),
      hidden_layer_output(perceptrons, 0) {
    initializeWeights();
  }

  // Run the model: get inputs, compute dot products, apply thresholding
  void run() {
    input_user = getBinaryInputs();

    // Compute dot product between input and weight matrix
    for (int j = 0; j < num_perceptrons; j++) {
      for (int i = 0; i < num_inputs; i++) {
        hidden_layer_sum[j] += input_user[i] * weight_matrix[i][j];
      }
    }

    // Thresholding the hidden layer with theta = 1
    int theta_hidden = 1;
    for (int j = 0; j < num_perceptrons; j++) {
      hidden_layer_output[j] = (hidden_layer_sum[j] >= theta_hidden) ? 1 : 0;
    }

    // Calculate the sum of thresholded outputs
    int output_layer_sum = 0;
    for (int i = 0; i < num_perceptrons; i++) {
      output_layer_sum += hidden_layer_output[i];
    }
    cout << "\nJumlah Total Elemen Setelah Thresholding output layer: " << output_layer_sum << endl;

    // Apply final thresholding based on majority activation
    int threshold_output = num_perceptrons / 2;
    int final_output = (output_layer_sum >= threshold_output) ? 1 : 0;

    cout << "\nHasil setelah Thresholding output layer:" << endl;
    cout << final_output << endl;
  }
};

int main() {
  int num_inputs = 11;
  int num_perceptrons = 2048;  // 2^11
  PerceptronModel model(num_inputs, num_perceptrons);
  model.run();
  return 0;
}