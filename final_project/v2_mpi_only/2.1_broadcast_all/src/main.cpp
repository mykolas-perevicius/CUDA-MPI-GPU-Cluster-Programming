// final_project/v2_mpi_only/2.1_broadcast_all/src/main.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm> // for std::min
#include <iomanip>   // For std::fixed, std::setprecision if needed
#include <chrono>
#include "../include/alexnet.hpp" // Include header which now defines inline helpers

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Variable Declarations ---
    int H = 227, W = 227, C = 3;
    LayerParams conv1, conv2;
    std::vector<float> input, finalOut;
    int finalH = -1, finalW = -1, finalC = -1;

    // 1) Common setup
    if (rank == 0) {
        // initialize input & params
        input.assign((size_t)H * W * C, 1.0f);
        conv1.K = 96; conv1.F = 11; conv1.S = 4; conv1.P = 0;
        conv1.F_pool = 3; conv1.S_pool = 2;
        conv1.weights.assign((size_t)conv1.K * C * conv1.F * conv1.F, 0.01f);
        conv1.biases.assign(conv1.K, 0.0f);
        conv2.K = 256; conv2.F = 5; conv2.S = 1; conv2.P = 2;
        conv2.F_pool = 3; conv2.S_pool = 2;
        conv2.N_lrn = 5; conv2.alpha = 1e-4f; conv2.beta = 0.75f; conv2.k_lrn = 2.0f;
        int C_conv2_input = conv1.K;
        conv2.weights.assign((size_t)conv2.K * C_conv2_input * conv2.F * conv2.F, 0.01f);
        conv2.biases.assign(conv2.K, 0.0f);

        // Calculate expected final dimensions (uses inline helpers from alexnet.hpp)
        int H1 = convOutDim(H, conv1.F, conv1.P, conv1.S);
        int W1 = convOutDim(W, conv1.F, conv1.P, conv1.S);
        H1 = poolOutDim(H1, conv1.F_pool, conv1.S_pool);
        W1 = poolOutDim(W1, conv1.F_pool, conv1.S_pool);
        int H2 = convOutDim(H1, conv2.F, conv2.P, conv2.S);
        int W2 = convOutDim(W1, conv2.F, conv2.P, conv2.S);
        H2 = poolOutDim(H2, conv2.F_pool, conv2.S_pool);
        W2 = poolOutDim(W2, conv2.F_pool, conv2.S_pool);
        finalH = H2; finalW = W2; finalC = conv2.K;
    }

    // 2) Broadcast EVERYTHING
    auto bcastInt = [&](int& x){ MPI_Bcast(&x,1,MPI_INT,0,MPI_COMM_WORLD); };
    bcastInt(H); bcastInt(W); bcastInt(C);
    bcastInt(conv1.K); bcastInt(conv1.F); bcastInt(conv1.S); bcastInt(conv1.P);
    bcastInt(conv1.F_pool); bcastInt(conv1.S_pool);
    bcastInt(conv2.K); bcastInt(conv2.F); bcastInt(conv2.S); bcastInt(conv2.P);
    bcastInt(conv2.F_pool); bcastInt(conv2.S_pool);
    bcastInt(conv2.N_lrn);
    bcastInt(finalH); bcastInt(finalW); bcastInt(finalC);

    if (rank != 0) {
        input.resize((size_t)H * W * C);
        conv1.weights.resize((size_t)conv1.K * C * conv1.F * conv1.F);
        conv1.biases.resize(conv1.K);
        int C_conv2_input = conv1.K;
        conv2.weights.resize((size_t)conv2.K * C_conv2_input * conv2.F * conv2.F);
        conv2.biases.resize(conv2.K);
    }
    float lrnArr[3];
    if (rank == 0) { lrnArr[0] = conv2.alpha; lrnArr[1] = conv2.beta; lrnArr[2] = conv2.k_lrn; }
    MPI_Bcast(lrnArr,3,MPI_FLOAT,0,MPI_COMM_WORLD);
    if (rank != 0) { conv2.alpha = lrnArr[0]; conv2.beta = lrnArr[1]; conv2.k_lrn = lrnArr[2]; }
    MPI_Bcast(input.data(),           static_cast<int>(input.size()),           MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv1.weights.data(),   static_cast<int>(conv1.weights.size()),   MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv1.biases.data(),    static_cast<int>(conv1.biases.size()),    MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv2.weights.data(),   static_cast<int>(conv2.weights.size()),   MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(conv2.biases.data(),    static_cast<int>(conv2.biases.size()),    MPI_FLOAT,0,MPI_COMM_WORLD);

    // --- Barrier before starting main computation ---
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_compute_start = std::chrono::high_resolution_clock::now();

    // 3) Perform Forward Pass (all ranks compute the full pass)
    alexnetForwardPassMPI(input, conv1, conv2, H, W, C, finalOut);

    // --- Barrier after main computation ---
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();

    // 4) Print results and time (ONLY from Rank 0)
    if (rank == 0) {
        // Calculate duration
        double duration_ms = std::chrono::duration<double, std::milli>(t_end - t_compute_start).count();

        // Print shape and sample values
        std::cout << "shape: " << finalH << "x" << finalW << "x" << finalC << std::endl;
        std::cout << "Sample values: ";
        int num_to_print = std::min((size_t)5, finalOut.size());
        // std::cout << std::fixed << std::setprecision(6); // Optional precision
        for (int i = 0; i < num_to_print; ++i) {
            std::cout << finalOut[i] << (i == num_to_print - 1 ? "" : " ");
        }
        std::cout << std::endl;

        // Print time in expected format
        std::cout << "Execution Time: " << duration_ms << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}