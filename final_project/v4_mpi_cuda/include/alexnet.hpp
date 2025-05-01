// === final_project/v4_mpi_cuda/include/alexnet.hpp ===
// (only the **new forward-declaration** added)
#ifndef ALEXNET_MPI_CUDA_HPP
#define ALEXNET_MPI_CUDA_HPP
#include <vector>
#include <cstddef>

struct LayerParams {
    std::vector<float> weights, biases;
    int K,F,S,P, F_pool,S_pool, N_lrn;
    float alpha,beta,k_lrn;
};


void alexnetForwardPassMPI_CUDA(const std::vector<float>& h_localInput,int localH,
                                int H,int W,int C,
                                const LayerParams& p1,const LayerParams& p2,
                                std::vector<float>& h_localOutput,
                                int rank,int size);

// -------------- NEW: single-GPU tile helper -----------------
void alexnetTileForwardCUDA(const float* d_input,
                            const LayerParams& p1,const LayerParams& p2,
                            int H,int W,int C,
                            float* d_output);

// size helpers (unchanged)
inline int convOutDim(int D,int F,int P,int S){
    return (S<=0||F>D+2*P) ? 0 : (D+2*P-F)/S+1;
}
inline int poolOutDim(int D,int F,int S){
    return (S<=0||D<F) ? 0 : (D-F)/S+1;
}
#endif
