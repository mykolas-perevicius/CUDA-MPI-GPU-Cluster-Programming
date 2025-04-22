#include <mpi.h>
#include <iostream>
#include <algorithm>     // std::copy_n
#include <numeric>
#include "../include/alexnet.hpp"
#include "../include/layers.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    /* ---------------- Rank‑0 initialises full problem ---------------- */
    int H = 227, W = 227, C = 3;
    LayerParams conv1, conv2;
    std::vector<float> input;
    std::vector<float> finalOut;

    if (rank == 0) {
        /* --- toy input (all ones) --- */
        input.resize(H*W*C, 1.0f);

        /* Conv1 + Pool1 params */
        conv1.K = 96; conv1.F = 11; conv1.S = 4; conv1.P = 0;
        conv1.F_pool = 3; conv1.S_pool = 2;
        size_t w1 = conv1.K * C * conv1.F * conv1.F;
        conv1.weights.assign(w1, 0.01f);
        conv1.biases.assign(conv1.K, 0.0f);

        /* Conv2 + Pool2 + LRN2 params */
        conv2.K = 256; conv2.F = 5; conv2.S = 1; conv2.P = 2;
        conv2.F_pool = 3; conv2.S_pool = 2;
        conv2.N_lrn = 5; conv2.alpha = 1e-4f; conv2.beta = 0.75f; conv2.k_lrn = 2.0f;
        size_t w2 = conv2.K * conv1.K * conv2.F * conv2.F;
        conv2.weights.assign(w2, 0.01f);
        conv2.biases.assign(conv2.K, 0.0f);
    }

    /* -------------------- Broadcast sizes -------------------- */
    auto bcastInt = [&](int& x){ MPI_Bcast(&x,1,MPI_INT,0,MPI_COMM_WORLD); };

    bcastInt(H);  bcastInt(W);  bcastInt(C);

    /* conv1 */
    bcastInt(conv1.K); bcastInt(conv1.F); bcastInt(conv1.S); bcastInt(conv1.P);
    bcastInt(conv1.F_pool); bcastInt(conv1.S_pool);

    /* conv2 */
    bcastInt(conv2.K); bcastInt(conv2.F); bcastInt(conv2.S); bcastInt(conv2.P);
    bcastInt(conv2.F_pool); bcastInt(conv2.S_pool);
    bcastInt(conv2.N_lrn);

    /* allocate on non‑root ranks */
    if (rank != 0) {
        conv1.biases.resize(conv1.K);
        conv1.weights.resize(conv1.K*C*conv1.F*conv1.F);
        conv2.biases.resize(conv2.K);
        conv2.weights.resize(conv2.K*conv1.K*conv2.F*conv2.F);
        input.resize(static_cast<size_t>(H)*W*C);
    }

    /* ---- broadcast the 3 LRN scalars safely ---- */
    float lrnParams[3];
    if (rank == 0) {
        lrnParams[0] = conv2.alpha;
        lrnParams[1] = conv2.beta;
        lrnParams[2] = conv2.k_lrn;
    }
    MPI_Bcast(lrnParams, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        conv2.alpha = lrnParams[0];
        conv2.beta  = lrnParams[1];
        conv2.k_lrn = lrnParams[2];
    }

    /* broadcast raw vectors */
    MPI_Bcast(input.data(),           input.size(),           MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv1.weights.data(),   conv1.weights.size(),   MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv1.biases.data(),    conv1.biases.size(),    MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv2.weights.data(),   conv2.weights.size(),   MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(conv2.biases.data(),    conv2.biases.size(),    MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* -------------------- Local forward pass -------------------- */
    std::vector<float> fullOutput;
    alexnetForwardPassMPI(input, conv1, conv2, H, W, C, fullOutput);

    /* final output dimensions after Conv2->ReLU2->Pool2->LRN2 */
    int H1 = convOutDim(H, conv1.F, conv1.P, conv1.S);
    H1 = convOutDim(H1, conv1.F_pool, 0, conv1.S_pool);
    int W1 = convOutDim(W, conv1.F, conv1.P, conv1.S);
    W1 = convOutDim(W1, conv1.F_pool, 0, conv1.S_pool);

    int H2 = convOutDim(H1, conv2.F, conv2.P, conv2.S);
    H2 = convOutDim(H2, conv2.F_pool, 0, conv2.S_pool);
    int W2 = convOutDim(W1, conv2.F, conv2.P, conv2.S);
    W2 = convOutDim(W2, conv2.F_pool, 0, conv2.S_pool);
    int C2 = conv2.K;

    /* ---------------- partition output rows across ranks ------------- */
    int rowsPerRank = H2 / size;
    int extras      = H2 % size;
    int startRow    = rank * rowsPerRank + std::min(rank, extras);
    int localRows   = rowsPerRank + (rank < extras ? 1 : 0);
    size_t sliceSize = static_cast<size_t>(localRows) * W2 * C2;

    std::vector<float> localSlice(sliceSize);
    for (int r = 0; r < localRows; ++r)
    {
        size_t globalRow = static_cast<size_t>(startRow + r);
        size_t srcOffset = globalRow * W2 * C2;
        std::copy_n(fullOutput.data() + srcOffset, W2 * C2,
                    localSlice.data() + static_cast<size_t>(r) * W2 * C2);
    }

    /* gather counts / displs (rank 0) */
    std::vector<int> recvCounts(size), displs(size);
    int localCount = static_cast<int>(sliceSize);
    MPI_Gather(&localCount, 1, MPI_INT,
               recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i-1] + recvCounts[i-1];
        finalOut.resize(static_cast<size_t>(H2)*W2*C2);
    }

    MPI_Gatherv(localSlice.data(), localCount, MPI_FLOAT,
                finalOut.data(), recvCounts.data(), displs.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // *** OUTPUT MODIFICATION for script compatibility ***
        std::cout << "shape: " << H2 << "x" << W2 << "x" << C2 << std::endl;
        std::cout << "Sample values: ";
        for (size_t i = 0; i < 5 && i < finalOut.size(); ++i)
            std::cout << finalOut[i] << (i == std::min(finalOut.size(), size_t(5)) - 1 ? "" : " ");
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
