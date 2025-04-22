#include <algorithm>
#include <cmath>
#include <limits>
#include "../include/layers.hpp"

inline size_t idx3D(int h,int w,int c,int W,int C){
    return (static_cast<size_t>(h)*W + w)*C + c;
}

void serialConvLayer(std::vector<float>& out,
                     const std::vector<float>& in,
                     const LayerParams& p,
                     int H, int W, int C)
{
    int Ho = convOutDim(H, p.F, p.P, p.S);
    int Wo = convOutDim(W, p.F, p.P, p.S);
    for (int k = 0; k < p.K; ++k) {
        for (int ho = 0; ho < Ho; ++ho) {
            for (int wo = 0; wo < Wo; ++wo) {
                float sum = p.biases[k];
                for (int c = 0; c < C; ++c) {
                    for (int fh = 0; fh < p.F; ++fh) {
                        for (int fw = 0; fw < p.F; ++fw) {
                            int hi = ho*p.S - p.P + fh;
                            int wi = wo*p.S - p.P + fw;
                            if (hi<0||hi>=H||wi<0||wi>=W) continue;
                            sum += in[idx3D(hi,wi,c,W,C)] *
                                   p.weights[(((k*C + c)*p.F + fh)*p.F) + fw];
                        }
                    }
                }
                out[idx3D(ho,wo,k,Wo,p.K)] = sum;
            }
        }
    }
}

void serialReluLayer(std::vector<float>& data)
{
    for (auto &v : data) v = std::max(0.0f, v);
}

void serialMaxPoolLayer(std::vector<float>& out,
                        const std::vector<float>& in,
                        int H, int W, int C,
                        int F_pool, int S_pool)
{
    int Ho = convOutDim(H, F_pool, 0, S_pool);
    int Wo = convOutDim(W, F_pool, 0, S_pool);
    for (int h = 0; h < Ho; ++h) {
        for (int w = 0; w < Wo; ++w) {
            for (int c = 0; c < C; ++c) {
                float mx = -std::numeric_limits<float>::infinity();
                for (int fh = 0; fh < F_pool; ++fh) {
                    for (int fw = 0; fw < F_pool; ++fw) {
                        int hi = h*S_pool + fh;
                        int wi = w*S_pool + fw;
                        mx = std::max(mx, in[idx3D(hi,wi,c,W,C)]);
                    }
                }
                out[idx3D(h,w,c,Wo,C)] = mx;
            }
        }
    }
}

void serialLRNLayer(std::vector<float>& out,
                    const std::vector<float>& in,
                    int H, int W, int C,
                    int N, float alpha, float beta, float k)
{
    int half = N/2;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                float sumSq = 0.0f;
                for (int i = std::max(0,c-half); i <= std::min(C-1,c+half); ++i) {
                    float v = in[idx3D(h,w,i,W,C)];
                    sumSq += v*v;
                }
                float denom = std::pow(k + alpha * sumSq / N, beta);
                out[idx3D(h,w,c,W,C)] =
                    in[idx3D(h,w,c,W,C)] / denom;
            }
        }
    }
}
