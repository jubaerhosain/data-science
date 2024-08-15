#include <iostream>
#include <vector>
#include <cmath>

// Define a 2D vector type for simplicity
template <typename T>
using Matrix = std::vector<std::vector<T>>;

// Activation function (ReLU)
double relu(double x) {
    return std::max(0.0, x);
}

// Convolution operation
Matrix<double> convolve(const Matrix<double>& input, const Matrix<double>& kernel) {
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int kernelSize = kernel.size();

    int outputSize = inputSize - kernelSize + 1;
    Matrix<double> output(outputSize, std::vector<double>(outputSize, 0.0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    output[i][j] += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = relu(output[i][j]); // Apply ReLU activation
        }
    }

    return output;
}

// Max pooling operation
Matrix<double> maxPool(const Matrix<double>& input, int poolSize) {
    int inputHeight = input.size();
    int inputWidth = input[0].size();
    int outputSize = inputSize / poolSize;

    Matrix<double> output(outputSize, std::vector<double>(outputSize, 0.0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            double maxValue = 0.0;
            for (int pi = 0; pi < poolSize; ++pi) {
                for (int pj = 0; pj < poolSize; ++pj) {
                    maxValue = std::max(maxValue, input[i * poolSize + pi][j * poolSize + pj]);
                }
            }
            output[i][j] = maxValue;
        }
    }

    return output;
}

// Fully connected layer
Matrix<double> fullyConnected(const Matrix<double>& input, const Matrix<double>& weights) {
    int inputSize = input.size();
    int outputSize = weights[0].size();

    Matrix<double> output(inputSize, std::vector<double>(outputSize, 0.0));

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int k = 0; k < input[0].size(); ++k) {
                output[i][j] += input[i][k] * weights[k][j];
            }
            output[i][j] = relu(output[i][j]); // Apply ReLU activation
        }
    }

    return output;
}

int main() {
    // Example input data (28x28 image)
    Matrix<double> input = {{...}, {...}, ...};

    // Example convolutional kernel (3x3)
    Matrix<double> kernel = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

    // Apply convolution
    Matrix<double> convResult = convolve(input, kernel);

    // Apply max pooling (2x2)
    Matrix<double> pooledResult = maxPool(convResult, 2);

    // Example fully connected layer weights
    Matrix<double> fcWeights = {{...}, {...}, ...};

    // Apply fully connected layer
    Matrix<double> fcResult = fullyConnected(pooledResult, fcWeights);

    // Display the final result
    for (const auto& row : fcResult) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
