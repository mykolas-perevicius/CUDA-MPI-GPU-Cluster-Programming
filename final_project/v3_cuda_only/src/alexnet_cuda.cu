#include <cstdio>         // for fprintf, stderr
#include <cstdlib>        // for exit
#include <cmath>          // for powf
#include <cuda_runtime.h>
#include <vector>
#include "../include/alexnet.hpp"
#include "../include/layers.hpp"

// Helper macro for checking CUDA errors
#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t err = call;                                 \
    if(err != cudaSuccess) {                                \
      std::fprintf(stderr,                                  \
        "CUDA error %s:%d '%s'\n",                         \
        __FILE__, __LINE__, cudaGetErrorString(err));       \
      std::exit(1);                                         \
    }                                                       \
  } while(0)

// Runs the sequence of Conv1→ReLU→Pool1→Conv2→ReLU→Pool2→LRN2 on device
void alexnetForwardPassCUDA(
    const std::vector<float>& input_host,
    const LayerParams& p1,
    const LayerParams& p2,
    int H,int W,int C,
    std::vector<float>& output_host)
{
    // Allocate device buffers
    float *d_input, *d_c1out, *d_p1out,
          *d_c2out, *d_p2out, *d_l2out;
    float *d_w1,*d_b1,*d_w2,*d_b2;

    // Conv1 dims
    int Hc1 = (H + 2*p1.P - p1.F)/p1.S + 1;
    int Wc1 = (W + 2*p1.P - p1.F)/p1.S + 1;
    int Hp1 = (Hc1 - p1.F_pool)/p1.S_pool + 1;
    int Wp1 = (Wc1 - p1.F_pool)/p1.S_pool + 1;

    // Conv2 dims
    int Hc2 = (Hp1 + 2*p2.P - p2.F)/p2.S + 1;
    int Wc2 = (Wp1 + 2*p2.P - p2.F)/p2.S + 1;
    int Hp2 = (Hc2 - p2.F_pool)/p2.S_pool + 1;
    int Wp2 = (Wc2 - p2.F_pool)/p2.S_pool + 1;

    // Sizes
    size_t in_sz   = (size_t)H*W*C;
    size_t c1_sz   = (size_t)Hc1*Wc1*p1.K;
    size_t p1_sz   = (size_t)Hp1*Wp1*p1.K;
    size_t c2_sz   = (size_t)Hc2*Wc2*p2.K;
    size_t p2_sz   = (size_t)Hp2*Wp2*p2.K;
    size_t l2_sz   = p2_sz;

    CUDA_CHECK(cudaMalloc(&d_input, in_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1out, c1_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1out, p1_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2out, c2_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2out, p2_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_l2out, l2_sz * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w1, p1.weights.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, p1.biases .size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, p2.weights.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, p2.biases .size()*sizeof(float)));

    // Copy host→device
    CUDA_CHECK(cudaMemcpy(d_input, input_host.data(),   in_sz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1,   p1.weights.data(),    p1.weights.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1,   p1.biases.data(),     p1.biases .size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2,   p2.weights.data(),    p2.weights.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2,   p2.biases.data(),     p2.biases .size()*sizeof(float), cudaMemcpyHostToDevice));

    // 1) Conv1→ReLU→Pool1
    cudaConvLayer(d_c1out, d_input, d_w1, d_b1, H, W, C, p1.K, p1.F, p1.S, p1.P);
    cudaReluLayer(d_c1out, (int)c1_sz);
    cudaMaxPoolLayer(d_p1out, d_c1out, Hc1, Wc1, p1.K, p1.F_pool, p1.S_pool);

    // 2) Conv2→ReLU→Pool2
    cudaConvLayer(d_c2out, d_p1out, d_w2, d_b2, Hp1, Wp1, p1.K, p2.K, p2.F, p2.S, p2.P);
    cudaReluLayer(d_c2out, (int)c2_sz);
    cudaMaxPoolLayer(d_p2out, d_c2out, Hc2, Wc2, p2.K, p2.F_pool, p2.S_pool);

    // 3) LRN2
    cudaLRNLayer(d_l2out, d_p2out, Hp2, Wp2, p2.K, p2.N_lrn, p2.alpha, p2.beta, p2.k_lrn);

    // Copy result back
    output_host.resize(l2_sz);
    CUDA_CHECK(cudaMemcpy(output_host.data(), d_l2out, l2_sz*sizeof(float), cudaMemcpyDeviceToHost));

    // Free
    cudaFree(d_input);
    cudaFree(d_c1out); cudaFree(d_p1out);
    cudaFree(d_c2out); cudaFree(d_p2out); cudaFree(d_l2out);
    cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2);
}
