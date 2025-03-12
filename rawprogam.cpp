#include <iostream>
#include <vector>

using namespace std;

// Function untuk mendapatkan input biner dari pengguna
vector<int> get_binary_inputs(int size) {
  vector<int> inputs;
  int         value, count = 0;
  cout << "Enter " << size << " binary digits (0/1): " << endl;
  for (int i = 0; i < size; i++) {
    cin >> value;
    inputs.push_back(value % 2);
    cout << "Input " << i + 1 << ": " << inputs[i] << endl;
    if (value == 1)
      count++;
  }
  cout << "Input 1 is " << count << endl;

  return inputs;
}

int main() {
  // Jumlah input dan perceptron
  int num_inputs      = 11;
  int num_perceptrons = 2048;  // 2^11

  // Inisialisasi matriks bobot (menggunakan vektor 2D)
  vector<vector<int>> weight_matrix(num_inputs, vector<int>(num_perceptrons, 0));

  // Loop untuk mengisi matriks bobot dengan pola +1 dan -1
  for (int i = 0; i < num_inputs; i++) {         // Iterasi untuk setiap input
    int block_size   = 1 << i;                   // Ukuran blok yang berulang (2^i)
    int weight_value = -1;                       // Mulai dengan -1
    for (int j = 0; j < num_perceptrons; j++) {  // Iterasi untuk setiap perceptron
      weight_matrix[i][j] = weight_value;
      if ((j + 1) % block_size == 0) {  // Ubah tanda setiap block_size
        weight_value *= -1;
      }
    }
  }

  // Ambil input dari pengguna
  vector<int> input_user = get_binary_inputs(num_inputs);

  // Hitung hasil dot product antara input dan bobot
  vector<int> hidden_layer_sum(num_perceptrons, 0);
  for (int j = 0; j < num_perceptrons; j++) {
    for (int i = 0; i < num_inputs; i++) {
      hidden_layer_sum[j] += input_user[i] * weight_matrix[i][j];
    }
  }

  // Thresholding dengan ambang batas (theta = 1)
  int         theta_hidden = 1;
  vector<int> hidden_layer_output(num_perceptrons, 0);
  for (int j = 0; j < num_perceptrons; j++) {
    hidden_layer_output[j] = (hidden_layer_sum[j] >= theta_hidden) ? 1 : 0;
  }

  // Jumlah total elemen hasil thresholding di hidden layer
  int output_layer_sum = 0;
  for (int i = 0; i < num_perceptrons; i++) {
    output_layer_sum += hidden_layer_output[i];
  }
  cout << "\nJumlah Total Elemen Setelah Thresholding output layer: " << output_layer_sum << endl;

  // *Perbaikan pada thresholding final output*
  // Daripada menggunakan nilai tetap (1024), kita gunakan aturan berdasarkan mayoritas aktivasi
  int threshold_output = num_perceptrons / 2;  // Misalnya 50% dari total perceptron
  int final_output     = (output_layer_sum >= threshold_output) ? 1 : 0;

  cout << "\nHasil setelah Thresholding output layer:" << endl;
  cout << final_output << endl;

  return 0;
}