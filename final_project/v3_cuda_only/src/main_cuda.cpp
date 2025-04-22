#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../include/alexnet.hpp"

// CPU helper for computing output dims
inline int convOutDim(int D, int F, int P, int S) {
    return (D + 2*P - F) / S + 1;
}

int main() {
    // 1) Setup identical to V1
    int H=227, W=227, C=3;
    LayerParams conv1, conv2;
    std::vector<float> input(H*W*C,1.0f), output;

    conv1.K=96; conv1.F=11; conv1.S=4; conv1.P=0;
    conv1.F_pool=3; conv1.S_pool=2;
    conv1.weights.assign(conv1.K*C*conv1.F*conv1.F,0.01f);
    conv1.biases.assign(conv1.K,0.0f);

    conv2.K=256; conv2.F=5; conv2.S=1; conv2.P=2;
    conv2.F_pool=3; conv2.S_pool=2;
    conv2.N_lrn=5; conv2.alpha=1e-4f; conv2.beta=0.75f; conv2.k_lrn=2.0f;
    conv2.weights.assign(conv2.K*conv1.K*conv2.F*conv2.F,0.01f);
    conv2.biases.assign(conv2.K,0.0f);

    // 2) Run on GPU
    auto t0 = std::chrono::high_resolution_clock::now();
    alexnetForwardPassCUDA(input, conv1, conv2, H, W, C, output);
    auto t1 = std::chrono::high_resolution_clock::now();

    // 3) Print timing & sample
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout<<"AlexNet CUDA Forward Pass completed in "<<ms<<" ms\n";

    std::cout<<"Final Output (first 10 values):";
    for(int i=0;i<10 && i<(int)output.size();++i)
        std::cout<<" "<<output[i];
    std::cout<<"\n";

    return 0;
}
