#ifndef CONV_LAYER_CUH
#define CONV_LAYER_CUH

// Host-side function to launch the tiled convolution kernel.
// h_input  : Host pointer for input data (size: C*H*W)
// h_kernel : Host pointer for the kernel weights (size: K*C*R*S)
// h_bias   : Host pointer for bias (size: K)
// h_output : Host pointer for output (size: K*outH*outW)
// C, H, W  : Number of input channels, height, and width.
// K        : Number of filters (output channels)
// R, S     : Kernel dimensions (rows and columns)
// stride   : Convolution stride.
void runConvolution(const float* h_input,
                    const float* h_kernel,
                    const float* h_bias,
                    float* h_output,
                    int C, int H, int W,
                    int K, int R, int S,
                    int stride);

#endif // CONV_LAYER_CUH
