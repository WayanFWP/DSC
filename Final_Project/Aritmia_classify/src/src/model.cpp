#include "model.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <random>

Model::Model(int inputSize, int hiddenSize, int outputSize, double lr)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(lr)
{
    std::srand(std::time(0));

    W1 = std::vector<std::vector<double>>(hiddenSize, std::vector<double>(inputSize));
    b1 = std::vector<double>(hiddenSize);
    W2 = std::vector<std::vector<double>>(outputSize, std::vector<double>(hiddenSize));
    b2 = std::vector<double>(outputSize);

    // Random init
    for (auto &row : W1)
        for (auto &val : row)
            val = (double(std::rand()) / RAND_MAX - 0.5) * 2;

    for (auto &row : W2)
        for (auto &val : row)
            val = (double(std::rand()) / RAND_MAX - 0.5) * 2;
}

double Model::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double Model::sigmoid_derivative(double x)
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> Model::softmax(const std::vector<double> &x)
{
    double maxVal = *std::max_element(x.begin(), x.end());
    std::vector<double> exps(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        exps[i] = std::exp(x[i] - maxVal);
        sum += exps[i];
    }
    for (double &val : exps)
        val /= sum;
    return exps;
}

double Model::dot(const std::vector<double> &a, const std::vector<double> &b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

void Model::add_to(std::vector<double> &a, const std::vector<double> &b, double scale)
{
    for (size_t i = 0; i < a.size(); ++i)
        a[i] += b[i] * scale;
}

void Model::add_to(std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b, double scale)
{
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a[i].size(); ++j)
            a[i][j] += b[i][j] * scale;
}

void Model::train(const std::vector<std::vector<double>> &X,
                  const std::vector<std::vector<double>> &Y,
                  int epochs)
{
    size_t n = X.size();
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalMSE = 0;

        for (size_t i = 0; i < n; ++i)
        {
            const auto &x = X[i];
            const auto &y = Y[i];

            // ---- Forward ----
            std::vector<double> z1(hiddenSize), a1(hiddenSize);
            for (int h = 0; h < hiddenSize; ++h)
            {
                z1[h] = dot(W1[h], x) + b1[h];
                a1[h] = sigmoid(z1[h]);
            }

            std::vector<double> z2(outputSize), y_hat(outputSize);
            for (int o = 0; o < outputSize; ++o)
                z2[o] = dot(W2[o], a1) + b2[o];
            y_hat = softmax(z2);

            // ---- Loss ----
            std::vector<double> error(outputSize);
            for (int o = 0; o < outputSize; ++o)
            {
                error[o] = y_hat[o] - y[o];
                totalMSE += 0.5 * error[o] * error[o];
            }

            // ---- Backprop ----
            std::vector<std::vector<double>> dW2(outputSize, std::vector<double>(hiddenSize));
            std::vector<double> db2(outputSize);
            for (int o = 0; o < outputSize; ++o)
            {
                db2[o] = error[o];
                for (int h = 0; h < hiddenSize; ++h)
                    dW2[o][h] = error[o] * a1[h];
            }

            std::vector<double> dA1(hiddenSize, 0.0);
            for (int h = 0; h < hiddenSize; ++h)
                for (int o = 0; o < outputSize; ++o)
                    dA1[h] += W2[o][h] * error[o];

            std::vector<std::vector<double>> dW1(hiddenSize, std::vector<double>(inputSize));
            std::vector<double> db1(hiddenSize);
            for (int h = 0; h < hiddenSize; ++h)
            {
                double dZ1 = dA1[h] * sigmoid_derivative(z1[h]);
                db1[h] = dZ1;
                for (int j = 0; j < inputSize; ++j)
                    dW1[h][j] = dZ1 * x[j];
            }

            // ---- Update ----
            add_to(W2, dW2, -learningRate);
            add_to(b2, db2, -learningRate);
            add_to(W1, dW1, -learningRate);
            add_to(b1, db1, -learningRate);
        }
        int correct = 0;
        for (size_t i = 0; i < n; ++i)
        {
            int pred = predict(X[i]);
            int actual = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
            if (pred == actual)
                correct++;
        }
        double acc = 100.0 * correct / n;

        mseHistory.push_back(totalMSE / n);
        accuracyHistory.push_back(acc);
    }
}

std::vector<double> Model::predictProb(const std::vector<double> &input)
{
    std::vector<double> z1(hiddenSize), a1(hiddenSize);
    for (int h = 0; h < hiddenSize; ++h)
        a1[h] = sigmoid(dot(W1[h], input) + b1[h]);

    std::vector<double> z2(outputSize);
    for (int o = 0; o < outputSize; ++o)
        z2[o] = dot(W2[o], a1) + b2[o];

    return softmax(z2);
}

int Model::predict(const std::vector<double> &input)
{
    auto probs = predictProb(input);
    return std::max_element(probs.begin(), probs.end()) - probs.begin();
}

std::vector<double> Model::getMSEHistory() const
{
    return mseHistory;
}

std::vector<double> Model::getAccuracyHistory() const
{
    return accuracyHistory;
}

void Model::normalizeTrain(std::vector<std::vector<double>> &X, double &rrMin, double &rrMax, double &qrsMin, double &qrsMax)
{
    rrMin = qrsMin = 1e9;
    rrMax = qrsMax = -1e9;

    for (const auto &x : X)
    {
        rrMin = std::min(rrMin, x[0]);
        rrMax = std::max(rrMax, x[0]);
        qrsMin = std::min(qrsMin, x[1]);
        qrsMax = std::max(qrsMax, x[1]);
    }

    for (auto &x : X)
    {
        x[0] = (x[0] - rrMin) / (rrMax - rrMin + 1e-9);
        x[1] = (x[1] - qrsMin) / (qrsMax - qrsMin + 1e-9);
    }
}

int Model::classifyArrhythmia(double rr, double qrsd)
{
    if (rr >= 0.6 && rr <= 1.2 && qrsd >= 80 && qrsd <= 100)
        return 0; // Normal
    else if (rr < 0.32)
        return 3; // Tachycardia
    else if (rr >= 0.32 && rr < 0.60 && qrsd >= 80 && qrsd <= 100)
        return 2; // R-on-T
    else if (rr >= 0.32 && rr < 0.60 && qrsd > 100)
        return 6; // PVC
    else if (rr >= 0.6 && rr <= 1.2 && qrsd > 100)
        return 5; // Fusion
    else if (rr > 1.2 && rr <= 1.66)
        return 4; // Bradycardia
    else if (rr > 1.66)
        return 1; // Dropped
    return -1;
}