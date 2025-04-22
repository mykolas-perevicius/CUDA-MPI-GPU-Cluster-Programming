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
void alexnetForwardPassMPI(std::vector<float>& input,
                           const LayerParams& conv1,
                           const LayerParams& conv2,
                           int H, int W, int C,
                           std::vector<float>& output);

/* helper */
inline int convOutDim(int dim, int F, int P, int S)
{
    return (dim - F + 2 * P) / S + 1;
}
