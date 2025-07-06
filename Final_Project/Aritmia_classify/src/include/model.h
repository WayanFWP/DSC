#ifndef MODEL_H
#define MODEL_H

#include <vector>

class Model {
 public:
  //   Model(int inputSize, int hiddenSize, int outputSize, double lr);
  Model(int inputSize, int hiddenSize1, int hiddenSize2, int hiddenSize3, int hiddenSize4, int outputSize, double lr);
  ~Model() {}
  static int classifyArrhythmia(double rr, double qrsd);

  void train(const std::vector<std::vector<double>> &X, const std::vector<std::vector<double>> &Y, int epochs);

  int                 predict(const std::vector<double> &input);
  std::vector<double> predictProb(const std::vector<double> &input);
  std::vector<double> getMSEHistory() const;
  std::vector<double> getAccuracyHistory() const;
  void                normalizeTrain(std::vector<std::vector<double>> &X, double &rrMin, double &rrMax, double &qrsMin, double &qrsMax);

 private:
//   int    inputSize;
//   int    hiddenSize;
  int inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize;
//   int    outputSize;
  double learningRate;

  std::vector<std::vector<double>> W1;
  std::vector<double>              b1;
  std::vector<std::vector<double>> W2;
  std::vector<double>              b2;
  std::vector<std::vector<double>> W3, W4, W5;
  std::vector<double>              b3, b4, b5;

  std::vector<double> mseHistory;
  std::vector<double> accuracyHistory;

  double              sigmoid(double x);
  double              sigmoid_derivative(double x);
  std::vector<double> softmax(const std::vector<double> &x);

  double dot(const std::vector<double> &a, const std::vector<double> &b);
  void   add_to(std::vector<double> &a, const std::vector<double> &b, double scale = 1.0);
  void   add_to(std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, double scale = 1.0);
};

#endif  // MODEL_H
