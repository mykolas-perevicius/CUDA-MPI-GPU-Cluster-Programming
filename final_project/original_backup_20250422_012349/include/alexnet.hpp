#ifndef ALEXNET_HPP
#define ALEXNET_HPP

// Function prototype for the AlexNet forward pass.
// This performs a dummy inference using custom CUDA kernels.
void alexnetForward(const float* input, float* output, int inputSize, int outputSize);

#endif // ALEXNET_HPP
