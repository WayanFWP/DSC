#include <iostream>
#include <vector>

using namespace std;

class NeuralNetwork {
 private:
  int                    num_inputs;
  int                    num_perceptrons;
  vector<vector<double>> weight_matrix;

  void initialize_weights() {
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

  int compute_final_output(const vector<int> &hidden_layer_output) {
    int output_layer_sum = 0;
    for (int i = 0; i < num_perceptrons; i++) {
      output_layer_sum += hidden_layer_output[i];
    }
    cout << "Active perceptrons: " << output_layer_sum << endl;
    int threshold_output = num_perceptrons / 2;
    return (output_layer_sum >= threshold_output) ? 1 : 0;
  }

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
    cout << "output is " << final_output << endl;
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
  if (choice == 'y' || choice == 'Y') nn.inputUser();
  else nn.testcase();

  return 0;
}