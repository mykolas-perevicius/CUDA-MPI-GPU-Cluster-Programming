// final_project/v4_mpi_cuda/src/main_mpi_cuda.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate
#include <chrono>
#include <algorithm> // For std::min
#include <cmath>     // For std::ceil, std::max
#include <cuda_runtime.h>
#include <limits.h> // For INT_MAX

#include "../include/alexnet.hpp" // Includes LayerParams, forward pass prototype, dim helpers

// Simple helper to check CUDA calls from host code if needed (mainly used in .cu files)
// In this .cpp file, major CUDA errors are more likely caught by MPI aborts triggered in alexnet_mpi_cuda.cu
inline void hostCheckCUDA(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        // Ensure only one rank prints the error before aborting
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
             fprintf(stderr, "CUDA Error in host code at %s:%d - %s\n", file, line, cudaGetErrorString(err));
             fflush(stderr);
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Use MPI_Abort for coordinated exit
    }
}
#define HOST_CUDA_CHECK(call) { hostCheckCUDA((call), __FILE__, __LINE__); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- 1. Root node initialization ---
    int H = 227, W = 227, C = 3; // Standard AlexNet input - H is now crucial and broadcast
    LayerParams p1, p2;
    std::vector<float> h_inputData; // Full input data on host (rank 0 only)
    std::vector<float> h_finalOutput; // Gathered final output on host (rank 0 only)
    int finalH = -1, finalW = -1, finalC = -1; // Expected final dimensions

    if (rank == 0) {
        std::cout << "--- AlexNet MPI+CUDA (V4 - Host Staging) ---" << std::endl;
        std::cout << "Initializing data and parameters on Rank 0..." << std::endl;

        h_inputData.assign((size_t)H * W * C, 1.0f);
        p1.K = 96; p1.F = 11; p1.S = 4; p1.P = 0;
        p1.F_pool = 3; p1.S_pool = 2;
        if (p1.F % 2 == 0 || p1.F_pool % 2 == 0) { std::cerr << "Warning: Filter sizes should be odd for simpler F/2 halo calculation." << std::endl; }
        p1.weights.assign((size_t)p1.K * C * p1.F * p1.F, 0.01f);
        p1.biases.assign(p1.K, 0.0f);
        p2.K = 256; p2.F = 5; p2.S = 1; p2.P = 2;
        p2.F_pool = 3; p2.S_pool = 2;
        if (p2.F % 2 == 0 || p2.F_pool % 2 == 0) { std::cerr << "Warning: Filter sizes should be odd for simpler F/2 halo calculation." << std::endl; }
        p2.N_lrn = 5; p2.alpha = 1e-4f; p2.beta = 0.75f; p2.k_lrn = 2.0f;
        int C_conv2_input = p1.K;
        p2.weights.assign((size_t)p2.K * C_conv2_input * p2.F * p2.F, 0.01f);
        p2.biases.assign(p2.K, 0.0f);

        int Hc1 = convOutDim(H, p1.F, p1.P, p1.S); int Wc1 = convOutDim(W, p1.F, p1.P, p1.S);
        int Hp1 = poolOutDim(Hc1, p1.F_pool, p1.S_pool); int Wp1 = poolOutDim(Wc1, p1.F_pool, p1.S_pool);
        int Hc2 = convOutDim(Hp1, p2.F, p2.P, p2.S); int Wc2 = convOutDim(Wp1, p2.F, p2.P, p2.S);
        int Hp2 = poolOutDim(Hc2, p2.F_pool, p2.S_pool); int Wp2 = poolOutDim(Wc2, p2.F_pool, p2.S_pool);
        finalH = Hp2; finalW = Wp2; finalC = p2.K;

        if (finalH <= 0 || finalW <= 0 || finalC <=0) { std::cerr << "Error: Calculated final dimensions invalid (" << finalH << "x" << finalW << "x" << finalC << ")." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        std::cout << "Expected final output dimensions: H=" << finalH << ", W=" << finalW << ", C=" << finalC << std::endl;
        std::cout << "Initialization complete." << std::endl;
    }

    // --- 2. Broadcast essential parameters ---
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD); // *** H is now broadcast ***
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p1.K, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p1.F, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p1.S, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p1.P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p1.F_pool, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p1.S_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p2.K, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p2.F, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p2.S, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p2.P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p2.F_pool, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&p2.S_pool, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p2.N_lrn, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&finalH, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&finalW, 1, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(&finalC, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float lrn_params[3];
    if (rank == 0) { lrn_params[0] = p2.alpha; lrn_params[1] = p2.beta; lrn_params[2] = p2.k_lrn; }
    MPI_Bcast(lrn_params, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank != 0) { p2.alpha = lrn_params[0]; p2.beta = lrn_params[1]; p2.k_lrn = lrn_params[2]; }

    size_t w1_size = (size_t)p1.K * C * p1.F * p1.F; size_t b1_size = (size_t)p1.K;
    size_t w2_size = (size_t)p2.K * p1.K * p2.F * p2.F; size_t b2_size = (size_t)p2.K;
    if (rank != 0) { p1.weights.resize(w1_size); p1.biases.resize(b1_size); p2.weights.resize(w2_size); p2.biases.resize(b2_size); }
    if (w1_size > INT_MAX || b1_size > INT_MAX || w2_size > INT_MAX || b2_size > INT_MAX) { if (rank == 0) std::cerr << "Error: Parameter size exceeds INT_MAX." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Bcast(p1.weights.data(), static_cast<int>(w1_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p1.biases.data(), static_cast<int>(b1_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p2.weights.data(), static_cast<int>(w2_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p2.biases.data(), static_cast<int>(b2_size), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- 3. Scatter Input Data ---
    std::vector<int> sendCounts(size); std::vector<int> displacements(size);
    std::vector<float> h_localInput; int localH = 0; int rowSize = W * C;
    if (rowSize <= 0 && H > 0 && W > 0 && C > 0) { if (rank == 0) std::cerr << "Error: Calculated rowSize invalid." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank == 0) {
        int baseRows = H / size; int extraRows = H % size; int currentDisplacement = 0;
        for (int i = 0; i < size; ++i) {
            int rowsForRank = baseRows + (i < extraRows ? 1 : 0); sendCounts[i] = rowsForRank * rowSize;
            if (rowsForRank > 0 && rowSize > 0 && sendCounts[i] / rowsForRank != rowSize ) { std::cerr << "Error: Scatter count overflow rank " << i << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
            displacements[i] = currentDisplacement;
            if (sendCounts[i] < 0 || currentDisplacement < 0 || currentDisplacement > INT_MAX - sendCounts[i]) { std::cerr << "Error: Scatter displacement overflow rank " << i << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
            currentDisplacement += sendCounts[i];
        }
        if (currentDisplacement != H * W * C && H > 0) { std::cerr << "Error: Total scatter size mismatch." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    int rowsForMyRank; std::vector<int> rowsPerRank(size);
    if (rank == 0) { int baseRows = H / size; int extraRows = H % size; for(int i=0; i<size; ++i) rowsPerRank[i] = baseRows + (i < extraRows ? 1 : 0); }
    MPI_Scatter(rowsPerRank.data(), 1, MPI_INT, &rowsForMyRank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localH = rowsForMyRank;
    int localInputSize = localH * rowSize; if (localInputSize < 0) localInputSize = 0;
    if (localInputSize > 0) h_localInput.resize(localInputSize); else h_localInput.clear();
    MPI_Scatterv( (rank == 0 && !h_inputData.empty()) ? h_inputData.data() : nullptr, sendCounts.data(), displacements.data(), MPI_FLOAT, h_localInput.empty() ? nullptr : h_localInput.data(), localInputSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- 4. Select GPU and Run Forward Pass ---
    int deviceId = 0; int numDevices = 0; cudaError_t cuda_err = cudaGetDeviceCount(&numDevices);
    if (cuda_err != cudaSuccess) { if (rank == 0) fprintf(stderr, "CUDA error cudaGetDeviceCount: %s\n", cudaGetErrorString(cuda_err)); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (numDevices > 0) { deviceId = rank % numDevices; HOST_CUDA_CHECK(cudaSetDevice(deviceId)); /* Print info if needed */ }
    else { if (rank == 0) std::cerr << "Error: No CUDA devices found." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
    std::vector<float> h_localOutput;
    MPI_Barrier(MPI_COMM_WORLD); auto t_start = std::chrono::high_resolution_clock::now();

    // *** UPDATE THE CALL: Pass H ***
    alexnetForwardPassMPI_CUDA(h_localInput, localH, H, W, C, p1, p2, h_localOutput, rank, size);

    MPI_Barrier(MPI_COMM_WORLD); auto t_end = std::chrono::high_resolution_clock::now();

    // --- 5. Gather Final Results ---
    int localOutputSize = static_cast<int>(h_localOutput.size()); if (localOutputSize < 0) localOutputSize = 0;
    std::vector<int> recvCounts(size, 0); std::vector<int> recvDisplacements(size, 0);
    MPI_Gather(&localOutputSize, 1, MPI_INT, recvCounts.empty() ? nullptr : recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    long long totalFinalSize_ll = 0;
    if (rank == 0) {
        if (recvCounts.empty()) { std::cerr << "Error: recvCounts empty rank 0." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        recvDisplacements.resize(size); recvDisplacements[0] = 0;
        if (recvCounts[0] < 0) { std::cerr << "Error: Rank 0 neg recv count." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        totalFinalSize_ll = recvCounts[0];
        for (int i = 1; i < size; ++i) {
            if (recvCounts[i-1] < 0) { std::cerr << "Error: Rank " << i-1 << " neg recv count." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
            long long prevDisp_ll = recvDisplacements[i-1];
            if (prevDisp_ll > INT_MAX - recvCounts[i-1]) { std::cerr << "Error: Gather displ overflow rank " << i << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
            recvDisplacements[i] = prevDisp_ll + recvCounts[i-1];
            if (recvCounts[i] < 0) { std::cerr << "Error: Rank " << i << " neg recv count." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
            totalFinalSize_ll += recvCounts[i];
        }
        if (totalFinalSize_ll > h_finalOutput.max_size()) { std::cerr << "Error: Total gather size exceeds vector max_size." << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        if (totalFinalSize_ll < 0) totalFinalSize_ll = 0;
        h_finalOutput.resize(static_cast<size_t>(totalFinalSize_ll));
    }
    MPI_Gatherv( h_localOutput.empty() ? nullptr : h_localOutput.data(), localOutputSize, MPI_FLOAT, (rank == 0 && !h_finalOutput.empty()) ? h_finalOutput.data() : nullptr, recvCounts.empty() ? nullptr : recvCounts.data(), recvDisplacements.empty() ? nullptr : recvDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- 6. Final Output and Timing (Rank 0) ---
    if (rank == 0) {
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "AlexNet MPI+CUDA Forward Pass completed in " << duration_ms << " ms" << std::endl;
        int expectedTotalSize = finalH * finalW * finalC; size_t gatheredSize = h_finalOutput.size();
        if (finalH > 0) { std::cout << "Final Output Shape: " << finalH << "x" << finalW << "x" << finalC << std::endl; }
        else { std::cout << "Final Output Shape: Calculation resulted in non-positive expected dimensions." << std::endl; }
        if (finalH > 0 && gatheredSize != static_cast<size_t>(expectedTotalSize)) { std::cerr << "WARNING: Gathered size (" << gatheredSize << ") != expected (" << expectedTotalSize << "). Check logic." << std::endl; }
        else if (finalH <= 0) { std::cout << "Gathered total size: " << gatheredSize << " elements." << std::endl; }
        std::cout << "Final Output (first 10 values):";
        size_t num_to_print = std::min((size_t)10, h_finalOutput.size());
        for (size_t i = 0; i < num_to_print; ++i) { std::cout << " " << h_finalOutput[i]; }
        if (h_finalOutput.size() > 10) std::cout << " ...";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}