#include <iostream>
#include <cstdlib>
#include "conv_layer.cuh"

int main() {
    // Define test parameters for a convolution layer similar to AlexNet's conv1.
    int C = 3;            // 3 input channels (RGB)
    int H = 227;          // Input height
    int W = 227;          // Input width
    int K = 96;           // Number of filters (output channels)
    int R = 11;           // Kernel rows
    int S = 11;           // Kernel columns
    int stride = 4;       // Stride for convolution

    // Calculate output dimensions.
    int outH = (H - R) / stride + 1;
    int outW = (W - S) / stride + 1;
    int inputSize = C * H * W;
    int kernelSize = K * C * R * S;
    int biasSize = K;
    int outputSize = K * outH * outW;

    // Allocate and initialize host memory.
    float* h_input = new float[inputSize];
    float* h_kernel = new float[kernelSize];
    float* h_bias = new float[biasSize];
    float* h_output = new float[outputSize];

    // Initialize input: for testing, set all values to 1.0.
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = 1.0f;
    }
    // Initialize kernel: for testing, set all weights to 0.01.
    for (int i = 0; i < kernelSize; i++) {
        h_kernel[i] = 0.01f;
    }
    // Initialize biases to 0.
    for (int i = 0; i < biasSize; i++) {
        h_bias[i] = 0.0f;
    }

    // Run the convolution.
    runConvolution(h_input, h_kernel, h_bias, h_output, C, H, W, K, R, S, stride);

    // Print out the first 10 values of the output for verification.
    std::cout << "Convolution Test Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10 && i < outputSize; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Clean up.
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_bias;
    delete[] h_output;

    return 0;
}
