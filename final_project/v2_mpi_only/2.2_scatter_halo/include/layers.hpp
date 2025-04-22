#pragma once
#include <vector>
#include "alexnet.hpp"

/* ----- layer kernels (CPU, identical to V1 serial) ----- */
void serialConvLayer(std::vector<float>& out,
                     const std::vector<float>& in,
                     const LayerParams& p,
                     int H, int W, int C);

void serialReluLayer(std::vector<float>& data);

void serialMaxPoolLayer(std::vector<float>& out,
                        const std::vector<float>& in,
                        int H, int W, int C,
                        int F_pool, int S_pool);

void serialLRNLayer(std::vector<float>& out,
                    const std::vector<float>& in,
                    int H, int W, int C,
                    int N, float alpha, float beta, float k);
