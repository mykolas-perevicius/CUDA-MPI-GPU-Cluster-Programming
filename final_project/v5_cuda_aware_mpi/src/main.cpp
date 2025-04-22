// final_project/src/main.cpp
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "alexnet.hpp"
#include "mpi_helper.hpp"

static const int BASE_IMG_WIDTH  = 227;
static const int BASE_IMG_HEIGHT = 227;
static const int CHANNELS        = 3;

// After Block2, output dims: 256 x 13 x 13 = 43264
static const int FINAL_C_OUT = 256;
static const int FINAL_H_OUT = 13;
static const int FINAL_W_OUT = 13;
static const int FINAL_OUTPUT_SIZE_PER_IMAGE = FINAL_C_OUT * FINAL_H_OUT * FINAL_W_OUT;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int multiplier = 1;
    if (argc >= 2) {
        multiplier = std::atoi(argv[1]);
        if (multiplier < 1) {
            multiplier = 1;
        }
    }

    // For simplicity, use batch size 1 per rank.
    int N_per_rank = 1;
    int inputSize_per_rank = N_per_rank * CHANNELS * BASE_IMG_HEIGHT * BASE_IMG_WIDTH;
    int outputSize_per_rank = N_per_rank * FINAL_OUTPUT_SIZE_PER_IMAGE;

    if (rank == 0) {
        std::cout << "MPI world size: " << world_size << std::endl;
        std::cout << "Batch size per rank: " << N_per_rank << std::endl;
        std::cout << "Total input elements per rank: " << inputSize_per_rank << std::endl;
        std::cout << "Final output elements per rank (after LRN2): " << outputSize_per_rank << std::endl;
    }

    std::cout << "Process " << rank << " starting with multiplier=" << multiplier << std::endl;

    // Allocate host memory for input and output
    std::vector<float> h_inputData(inputSize_per_rank);
    std::vector<float> h_localOutput(outputSize_per_rank);

    // Initialize synthetic input data
    for (int i = 0; i < inputSize_per_rank; i++) {
        h_inputData[i] = static_cast<float>((rank + 1) + (i % 13)) / (10.0f + rank);
    }

    // ----- Initialize Conv1 and Conv2 weights (for simplicity, using placeholder initialization) -----
    // Conv1 parameters (assumed from previous block)
    size_t conv1_weights_count = 96UL * 3UL * 11UL * 11UL;
    size_t conv1_biases_count  = 96UL;
    std::vector<float> h_conv1_weights(conv1_weights_count);
    std::vector<float> h_conv1_biases(conv1_biases_count);
    for (size_t i = 0; i < conv1_weights_count; i++) {
        h_conv1_weights[i] = 0.01f + i * 0.001f;
    }
    std::fill(h_conv1_biases.begin(), h_conv1_biases.end(), 0.1f);

    // Conv2 parameters: 256 filters, input channels = 96 (output from Block1)
    size_t conv2_weights_count = 256UL * 96UL * 5UL * 5UL;
    size_t conv2_biases_count  = 256UL;
    std::vector<float> h_conv2_weights(conv2_weights_count);
    std::vector<float> h_conv2_biases(conv2_biases_count);
    for (size_t i = 0; i < conv2_weights_count; i++) {
        h_conv2_weights[i] = 0.001f + i * 0.0001f;
    }
    std::fill(h_conv2_biases.begin(), h_conv2_biases.end(), 0.2f);

    if (rank == 0) {
        std::cout << "Rank 0: Initialized Conv1 and Conv2 weights. Broadcasting..." << std::endl;
    }
    MPI_Bcast(h_conv1_weights.data(), conv1_weights_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_conv1_biases.data(),  conv1_biases_count,  MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_conv2_weights.data(), conv2_weights_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_conv2_biases.data(),  conv2_biases_count,  MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ----- Run the full AlexNet forward pass (Block1 and Block2) -----
    double start_t = MPI_Wtime();
    // alexnetForward now computes both Block1 and Block2,
    // Final output size is 256x13x13 = 43264 elements per image.
    alexnetForward(h_inputData.data(), h_localOutput.data(),
                   N_per_rank, CHANNELS, BASE_IMG_HEIGHT, BASE_IMG_WIDTH,
                   outputSize_per_rank);
    double end_t = MPI_Wtime();
    double elapsed = end_t - start_t;
    std::cout << "Process " << rank << " forward pass time (MPI_Wtime): " 
              << elapsed << " seconds." << std::endl;

    // Gather final outputs
    std::vector<float> gatheredOutputs;
    if (rank == 0) {
        gatheredOutputs.resize(world_size * outputSize_per_rank);
    }
    MPI_Gather(h_localOutput.data(), outputSize_per_rank, MPI_FLOAT,
               gatheredOutputs.data(), outputSize_per_rank, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Gathered final outputs (first 5 values per rank):" << std::endl;
        for (int r = 0; r < world_size; r++) {
            std::cout << "Process " << r << " output: ";
            for (int i = 0; i < 5 && i < outputSize_per_rank; i++) {
                std::cout << gatheredOutputs[r * outputSize_per_rank + i] << " ";
            }
            std::cout << "..." << std::endl;
        }
    }
    MPI_Finalize();
    return 0;
}
