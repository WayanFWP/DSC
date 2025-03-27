#include <iostream>

#include "MLP_EBPA.hpp"
#include "dataset.hpp"

using namespace std;

#define INPUT_SIZE 144
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 8
#define EPOCHS 350
#define LEARNING_RATE 0.1

int main() {
  MLP mlp(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);

  // Load dataset into memory dynamically
  vector<vector<vector<double>>> letters;
  vector<vector<vector<double>>> variations;
  load_dataset(letters, variations);

  vector<vector<double>> inputs;
  vector<vector<double>> targets;

  char label = 'A';
  for (size_t i = 0; i < letters.size(); ++i) {
    inputs.push_back(flatten(letters[i]));
    targets.push_back(get_target(label));

    inputs.push_back(flatten(variations[i * 2]));  // First variation
    targets.push_back(get_target(label));

    inputs.push_back(flatten(variations[i * 2 + 1]));  // Second variation
    targets.push_back(get_target(label));

    label++;
  }

  mlp.train(inputs, targets, EPOCHS);

  while (true) {
    auto promptToDisplayDataset = []() -> bool {
      char choice;
      std::cout << "Do you want to see the letter? (y/n): ";
      std::cin >> choice;
      return choice == 'y';
    };

    if (promptToDisplayDataset()) {
      display_dataset(letters);
    }

    int mode = 0;
    std::cout << "\n🤖 MLP Classifier\n"
              << "Enter 1 to test the model on /test directory\n"
              << "Enter 2 to input your own matrix\n"
              << "Your choice: ";
    std::cin >> mode;

    switch (mode) {
    case 2:
      predict_user_input(mlp);
      break;
    case 1:
    test_files(mlp);
    default:
      cout << "Invalid choice! Please enter 1 or 2." << endl;
    break;
    }
  }

  return 0;
}
