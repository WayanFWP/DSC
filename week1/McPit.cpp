#include <iostream>
#include <vector>

using namespace std;

class NeuralNetwork {
 private:
  int                    num_inputs;
  int                    num_perceptrons;
  vector<vector<double>> weight_matrix;
  vector<int>            input_user;
  vector<int>            hidden_layer_sum;
  vector<int>            hidden_layer_output;

  void initialize_weights() {
    weight_matrix.resize(num_inputs, vector<double>(num_perceptrons, 0));
    for (int i = 0; i < num_inputs; i++) {
      int block_size   = 1 << i;
      int weight_value = -1;
      for (int j = 0; j < num_perceptrons; j++) {
        weight_matrix[i][j] = weight_value;
        if ((j + 1) % block_size == 0) {
          weight_value *= -1;
        }
      }
    }
  }

 public:
  NeuralNetwork(int inputs, int perceptrons) : num_inputs(inputs), num_perceptrons(perceptrons) {
    weight_matrix.resize(num_inputs, vector<double>(num_perceptrons, 0.0));
    initialize_weights();
  }

  vector<int> userInput() {
    input_user.resize(num_inputs, 0);
    int inputs;
    int count = 0;
    cout << "Enter " << num_inputs << " binary digits (0/1): " << endl;
    for (int i = 0; i < num_inputs; i++) {
      cout << "Input " << i + 1 << ": ";
      cin >> inputs;
      input_user[i] = inputs % 2;
      if (inputs == 1)
        count++;
    }
    cout << "input 1 is " << count << endl;
    return input_user;
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
  void compile() {
    input_user = testcase();
    hidden_layer_sum.assign(num_perceptrons, 0);
    hidden_layer_output.assign(num_perceptrons, 0);

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
    int final_output     = (output_layer_sum >= threshold_output) ? 1 : 0;

    cout << "\nHasil setelah Thresholding output layer:" << endl;
    cout << final_output << endl;
  }
};

int main() {
  NeuralNetwork nn(11, 2048);
  nn.compile();

  return 0;
}