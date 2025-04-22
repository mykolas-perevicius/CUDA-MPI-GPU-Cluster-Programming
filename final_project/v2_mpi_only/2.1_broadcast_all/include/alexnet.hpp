#pragma once
#include <vector>

/* -------- Layer hyper‑parameters -------- */
struct LayerParams {
    std::vector<float> weights;   // flattened (K x C x F x F)
    std::vector<float> biases;    // length K
    int K = 0;    // # filters / output channels
    int F = 0;    // filter size
    int S = 1;    // stride
    int P = 0;    // padding

    /* pool */
    int F_pool  = 0;  // pool window
    int S_pool  = 1;  // pool stride

    /* local response norm */
    int   N_lrn = 0;
    float alpha = 0.0f;
    float beta  = 0.0f;
    float k_lrn = 1.0f;
};

/* --------------- API --------------- */
// This function needs to be defined in alexnet_mpi.cpp
void alexnetForwardPassMPI(std::vector<float>& input,
                           const LayerParams& conv1,
                           const LayerParams& conv2,
                           int H, int W, int C,
                           std::vector<float>& output);

/* -------- Helpers -------- */
// Defined inline so they can be included in multiple places without linker errors
inline int convOutDim(int dim, int F, int P, int S) {
    if (S <= 0) return 0; // Avoid division by zero
    return (dim - F + 2 * P) / S + 1;
}

inline int poolOutDim(int dim, int F, int S) {
    if (S <= 0) return 0; // Avoid division by zero
    return (dim - F) / S + 1; // Assumes P=0 for pooling
}