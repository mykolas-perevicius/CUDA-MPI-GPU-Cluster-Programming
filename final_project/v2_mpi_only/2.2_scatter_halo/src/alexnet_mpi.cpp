#include "../include/alexnet.hpp"
#include "../include/layers.hpp"

void alexnetForwardPassMPI(std::vector<float>& input,
                           const LayerParams& conv1,
                           const LayerParams& conv2,
                           int H, int W, int C,
                           std::vector<float>& output)
{
    int Hc1 = convOutDim(H, conv1.F, conv1.P, conv1.S);
    int Wc1 = convOutDim(W, conv1.F, conv1.P, conv1.S);
    std::vector<float> conv1Out(Hc1 * Wc1 * conv1.K);

    serialConvLayer(conv1Out, input, conv1, H, W, C);
    serialReluLayer(conv1Out);

    int Hp1 = convOutDim(Hc1, conv1.F_pool, 0, conv1.S_pool);
    int Wp1 = convOutDim(Wc1, conv1.F_pool, 0, conv1.S_pool);
    std::vector<float> pool1Out(Hp1 * Wp1 * conv1.K);
    serialMaxPoolLayer(pool1Out, conv1Out, Hc1, Wc1, conv1.K,
                       conv1.F_pool, conv1.S_pool);

    int Hc2 = convOutDim(Hp1, conv2.F, conv2.P, conv2.S);
    int Wc2 = convOutDim(Wp1, conv2.F, conv2.P, conv2.S);
    std::vector<float> conv2Out(Hc2 * Wc2 * conv2.K);
    serialConvLayer(conv2Out, pool1Out, conv2, Hp1, Wp1, conv1.K);
    serialReluLayer(conv2Out);

    int Hp2 = convOutDim(Hc2, conv2.F_pool, 0, conv2.S_pool);
    int Wp2 = convOutDim(Wc2, conv2.F_pool, 0, conv2.S_pool);
    std::vector<float> pool2Out(Hp2 * Wp2 * conv2.K);
    serialMaxPoolLayer(pool2Out, conv2Out, Hc2, Wc2, conv2.K,
                       conv2.F_pool, conv2.S_pool);

    output.resize(pool2Out.size());
    serialLRNLayer(output, pool2Out, Hp2, Wp2, conv2.K,
                   conv2.N_lrn, conv2.alpha, conv2.beta, conv2.k_lrn);
}
