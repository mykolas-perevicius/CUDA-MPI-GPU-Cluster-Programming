# CS485: GPU Cluster Programming (MPI+CUDA) - Comprehensive Repository

Welcome to Mykolas Perevicius's repository for CS485: GPU Cluster Programming at NJIT. This repository directory contains the final project code (AlexNet inference implementation), assignments, and supporting materials. Our objective is to develop robust, high-performance parallel programs using MPI, CUDA, and their combination for GPU clusters.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Target Environment](#target-environment)
- [Project Core: AlexNet Inference (Blocks 1 & 2)](#project-core-alexnet-inference-blocks-1--2)
  - [Problem Definition](#problem-definition)
  - [Input and Output](#input-and-output)
- [Staged Implementation Details (V1-V4)](#staged-implementation-details-v1-v4)
  - [V1: Serial CPU (Baseline)](#v1-serial-cpu-baseline)
    - [Goal and Paradigm](#goal-and-paradigm-v1)
    - [Key Implementation Strategy & Code Examples V1](#key-implementation-strategy--code-examples-v1)
    - [Key Files V1](#key-files-v1)
    - [Makefile Highlights V1](#makefile-highlights-v1)
  - [V2.2: MPI Only (Scatter+Halo)](#v22-mpi-only-scatterhalo)
    - [Goal and Paradigm V2.2](#goal-and-paradigm-v22)
    - [Key Implementation Strategy & Code Examples V2.2](#key-implementation-strategy--code-examples-v22)
    - [Key Files V2.2](#key-files-v22)
    - [Makefile Highlights V2.2](#makefile-highlights-v22)
  - [V3: CUDA Only (Single GPU)](#v3-cuda-only-single-gpu)
    - [Goal and Paradigm V3](#goal-and-paradigm-v3)
    - [Key Implementation Strategy & Code Examples V3](#key-implementation-strategy--code-examples-v3)
    - [Key Files V3](#key-files-v3)
    - [Makefile Highlights V3](#makefile-highlights-v3)
  - [V4: MPI+CUDA (Hybrid Host-Staging)](#v4-mpicuda-hybrid-host-staging)
    - [Goal and Paradigm V4](#goal-and-paradigm-v4)
    - [Key Implementation Strategy & Code Examples V4](#key-implementation-strategy--code-examples-v4)
    - [Key Files V4](#key-files-v4)
    - [Makefile Highlights V4](#makefile-highlights-v4)
- [Compilation and Execution Summary](#compilation-and-execution-summary)
- [Summary of Performance Findings and Conclusions](#summary-of-performance-findings-and-conclusions)
  - [Performance Highlights](#performance-highlights)
  - [Key Bottlenecks Identified](#key-bottlenecks-identified)
  - [Overall Lessons Learned](#overall-lessons-learned)
- [Future Work Highlights](#future-work-highlights)
- [Project Report](#project-report)

---

## Overview

This repository serves as a comprehensive workspace for CS485. The final project, demonstrating an implementation of AlexNet inference for the initial two blocks (Conv1 $\rightarrow$ ReLU $\rightarrow$ MaxPool1 and Conv2 $\rightarrow$ ReLU $\rightarrow$ MaxPool2 $\rightarrow$ LRN2), is the centerpiece. It leverages a data-parallel strategy over MPI ranks and explores different parallel programming paradigms. Automation scripts ensure reproducible builds, testing across various configurations, and final packaging for submission. The project aims not only to achieve parallelism but also to critically analyze the performance trade-offs and complexities inherent in each approach.

---

## Repository Structure

The repository is organized as follows:

-   **`final_project/`**: Contains the core source code for the AlexNet inference project, divided into subdirectories for each implementation version (V1, V2.1, V2.2, V3, V4).
    -   **`v1_serial/`**: Serial CPU implementation.
    -   **`v2_mpi_only/`**: MPI-only implementations.
        -   **`2.1_broadcast_all/`**: MPI with data broadcast.
        -   **`2.2_scatter_halo/`**: MPI with input scatter and halo exchange.
    -   **`v3_cuda_only/`**: CUDA-only single GPU implementation.
    -   **`v4_mpi_cuda/`**: Hybrid MPI+CUDA implementation (host-staging).
    -   **`data/`**: (Potentially) Input data, weights, etc. (though initialization is often programmatic in this project).
    -   **`docs/`**: Supplementary documentation, images used in reports/presentations.
    -   **`logs/`**: Directory for storing execution logs and performance results.
-   **`scripts/`**: Bash scripts for automating tasks like building, running tests, and packaging.
-   **`homeworks/`**: (If applicable) Solutions to course homework assignments.
-   **`README.md`**: This file, providing a comprehensive guide to the repository.
-   **Other files**: Configuration files (e.g., `.gitignore`, `shell.nix`), potentially the final PDF report.

---

## Target Environment

The project code is developed to be compatible with and is primarily benchmarked on the following target environment:
-   **Operating System:** Fedora 37
-   **Host Compiler:** GCC 12
-   **MPI Implementation:** Open MPI (version 4.1.x or similar, accessible via `mpicc`/`mpicxx`)
-   **GPU Toolkit:** NVIDIA CUDA 12.x (`nvcc` compiler)
-   Local development was often performed on NixOS with a similar toolchain, ensuring portability.

---

## Project Core: AlexNet Inference (Blocks 1 & 2)

### Problem Definition

The project focuses on implementing and parallelizing the inference (forward pass) for the first two computationally significant blocks of the AlexNet CNN. The sequence of layers is:
1.  **Block 1:** Convolution 1 (Conv1) $\rightarrow$ ReLU1 $\rightarrow$ Max Pooling 1 (Pool1)
2.  **Block 2:** Convolution 2 (Conv2) $\rightarrow$ ReLU2 $\rightarrow$ Max Pooling 2 (Pool2) $\rightarrow$ Local Response Normalization 2 (LRN2)

### Input and Output

-   **Input:** A single image represented as a 3D tensor of size 227x227x3 (Height x Width x Channels - RGB), with floating-point values. Total input elements: 154,587 floats.
-   **Output:** A 3D feature map tensor of size 13x13x256 (Height x Width x Channels), also with floating-point values. Total output elements: 43,264 floats.

Key operations include:
-   **Conv1:** 96 filters of 11x11x3, Stride 4.
-   **Conv2:** 256 filters of 5x5x96, Stride 1, Padding 2.

---

## Staged Implementation Details (V1-V4)

The project progressed through four distinct versions, each building upon the last or exploring a different parallel paradigm.

### V1: Serial CPU (Baseline)

#### Goal and Paradigm V1
To establish a functionally correct, single-threaded C++ implementation running on a single CPU core. This serves as the reference for correctness and the baseline for performance comparisons.

#### Key Implementation Strategy & Code Examples V1
All operations (convolution, ReLU, pooling, LRN) are implemented using standard C++ loops. Data is typically stored in `std::vector<float>`.

**Orchestration (`alexnet_serial.cpp`):**
The `alexnetForwardPass` function sequentially calls each layer function, passing data between them using intermediate `std::vector` buffers. It uses a double-buffering technique (`current_input`, `current_output` pointers swapping) to manage data flow.

```cpp
// In alexnet_serial.cpp
void alexnetForwardPass(
    std::vector<float>& input_data,
    const LayerParams& paramsConv1,
    const LayerParams& paramsConv2,
    int H, int W, int C)
{
    // ... setup, timing ...
    std::vector<float> buffer1, buffer2;
    std::vector<float>* current_input = &input_data;
    std::vector<float>* current_output = &buffer1;
    // ...
    // Block 1
    // Conv1
    serialConvLayer(*current_output, *current_input, paramsConv1.weights, /*...*/);
    std::swap(current_input, current_output); // Output of Conv1 becomes input for ReLU1
    // ReLU1
    serialReluLayer(*current_input); // In-place
    // Pool1
    serialMaxPoolLayer(*current_output, *current_input, /*...*/);
    std::swap(current_input, current_output);
    // ... Block 2 follows similarly ...
}
```

**Convolution Layer (`layers_serial.cpp`):**
A naive convolution implemented with nested loops.
```cpp
// In layers_serial.cpp
void serialConvLayer(
    std::vector<float>& output,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int H, int W, int C, // Input dimensions
    int K, // Num output channels (filters)
    int F, // Filter size
    int S, // Stride
    int P  // Padding
) {
    int Ho = (H - F + 2 * P) / S + 1;
    int Wo = (W - F + 2 * P) / S + 1;
    // For each output channel (filter k)
    for (int k = 0; k < K; ++k) {
        // For each output row (ho)
        for (int ho = 0; ho < Ho; ++ho) {
            // For each output column (wo)
            for (int wo = 0; wo < Wo; ++wo) {
                float sum = biases[k];
                // For each input channel (c)
                for (int c_in = 0; c_in < C; ++c_in) {
                    // For each filter row (fh)
                    for (int fh = 0; fh < F; ++fh) {
                        // For each filter column (fw)
                        for (int fw = 0; fw < F; ++fw) {
                            int hi = ho * S - P + fh; // Input row
                            int wi = wo * S - P + fw; // Input col
                            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                // input_idx maps (hi, wi, c_in)
                                // weight_idx maps (k, c_in, fh, fw)
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                output[output_idx] = sum; // output_idx maps (ho, wo, k)
            }
        }
    }
}```

#### Key Files V1
-   `v1_serial/src/main.cpp`
-   `v1_serial/src/alexnet_serial.cpp`
-   `v1_serial/src/layers_serial.cpp`
-   `v1_serial/include/alexnet.hpp`, `v1_serial/include/layers.hpp`

#### Makefile Highlights V1
Uses `g++` for compilation with standard C++11 flags and optimization (`-O3`).

### V2.2: MPI Only (Scatter+Halo)

#### Goal and Paradigm V2.2
To parallelize the serial CPU implementation across multiple CPU cores on potentially multiple machines using the Message Passing Interface (MPI). This version focuses on a scalable domain decomposition strategy (row-wise scatter of input feature maps) with halo exchange for boundary computations. SPMD (Single Program, Multiple Data) model.

#### Key Implementation Strategy & Code Examples V2.2
The root MPI rank (rank 0) initializes data and parameters. Parameters (weights, biases) are broadcast to all ranks. The input image's rows are scattered among MPI ranks. For convolutional layers, ranks that process boundary regions of their data slice need data from neighboring ranks ("halos"). This is achieved using non-blocking MPI sends/receives (`MPI_Isend`, `MPI_Irecv`) followed by `MPI_Waitall`. After local computation (Conv $\rightarrow$ ReLU $\rightarrow$ Pool), ranks perform an "asymmetric trim" of their output feature maps to remove rows affected by halo data from other ranks, especially after pooling layers change dimensions. Finally, rank 0 gathers the valid output slices.

**Main Logic (`v2_mpi_only/2.2_scatter_halo/src/main.cpp`):**
```cpp
// In v2_mpi_only/2.2_scatter_halo/src/main.cpp
// ... MPI_Init, rank, size ...
// 1. Rank 0 initializes full input & parameters
// 2. Broadcast parameters (weights, biases, layer dimensions) to all ranks

// 3. Scatter input image rows to all ranks
//    - Rank 0 calculates sendCounts and displacements for MPI_Scatterv
//    - Each rank receives its `localIn` (local slice of input rows)
std::vector<float> localIn(localCnt);
MPI_Scatterv(input.data(), sendCnt.data(), sendDisp.data(), MPI_FLOAT,
             localIn.data(), localCnt, MPI_FLOAT, 0, MPI_COMM_WORLD);
int localH = localCnt / (W * C); // Height of local slice

// 4. Halo Exchange for Conv1 (Conceptual - details vary based on padding (P) and filter size (F))
const int pad1_rows = conv1.F / 2; // Number of halo rows needed from neighbors
int slice1_elements = pad1_rows * W * C;
std::vector<float> topHalo_recv(slice1_elements), botHalo_recv(slice1_elements);
MPI_Request reqs; int req_count = 0;

if (rank > 0) { // If not the first rank, receive from rank-1 and send to rank-1
    MPI_Irecv(topHalo_recv.data(), slice1_elements, MPI_FLOAT, rank - 1, /*TAG_UP*/0, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(localIn.data(), slice1_elements, MPI_FLOAT, rank - 1, /*TAG_DOWN*/1, MPI_COMM_WORLD, &reqs[req_count++]);
}
if (rank < size - 1) { // If not the last rank, receive from rank+1 and send to rank+1
    MPI_Irecv(botHalo_recv.data(), slice1_elements, MPI_FLOAT, rank + 1, /*TAG_DOWN*/1, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(localIn.data() + (localH - pad1_rows) * W * C, slice1_elements, MPI_FLOAT, rank + 1, /*TAG_UP*/0, MPI_COMM_WORLD, &reqs[req_count++]);
}
MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

// 5. Construct padded input using localIn, topHalo_recv, botHalo_recv
//    Perform Conv1 -> ReLU1 -> Pool1 on this padded input
//    (Code for layer calls similar to V1, but on `padded_input1`)
//    Result in `pool1_intermediate_output`

// 6. Asymmetric Trim of `pool1_intermediate_output`
//    - Calculate how many rows at top/bottom of `pool1_intermediate_output`
//      are invalid due to halo propagation through Conv1/Pool1.
//    - Create `pool1_valid_output` by copying only the valid rows.
//    - This step is crucial and complex.

// 7. Halo Exchange for Conv2 (similar to step 4, using `pool1_valid_output`)
//    Construct `padded_input2`

// 8. Perform Conv2 -> ReLU2 -> Pool2 -> LRN2 on `padded_input2`
//    Result in `lrn2_intermediate_output`

// 9. Asymmetric Trim of `lrn2_intermediate_output` to get `localFinalOut`

// 10. Gather `localFinalOut` from all ranks to rank 0 using MPI_Gatherv
// ... MPI_Finalize ...
```

#### Key Files V2.2
-   `v2_mpi_only/2.2_scatter_halo/src/main.cpp`
-   `v2_mpi_only/2.2_scatter_halo/src/alexnet_mpi.cpp` (orchestrates layers for a given slice)
-   `v2_mpi_only/2.2_scatter_halo/src/layers_mpi.cpp` (serial layer implementations, same as V1)
-   `v2_mpi_only/2.2_scatter_halo/include/alexnet.hpp`, `layers.hpp`

#### Makefile Highlights V2.2
Uses `mpicxx` as the compiler, which handles MPI library linking.

### V3: CUDA Only (Single GPU)

#### Goal and Paradigm V3
To accelerate the AlexNet inference by porting the computationally intensive layer operations to run on a single NVIDIA GPU using CUDA. This leverages the GPU's SIMT (Single Instruction, Multiple Threads) architecture for massive parallelism.

#### Key Implementation Strategy & Code Examples V3
The host (CPU) code manages overall control, data initialization, and memory transfers between host RAM and GPU device memory. CUDA kernels are written for each layer (Conv, ReLU, Pool, LRN) to execute on the GPU.

**Host-Side Orchestration (`alexnet_cuda.cu`):**
The `alexnetForwardPassCUDA` function handles:
1.  Allocating GPU memory for input, output, intermediate feature maps, weights, and biases (`cudaMalloc`).
2.  Copying input data and parameters from host to device (`cudaMemcpyHostToDevice`).
3.  Launching a sequence of CUDA kernels for each layer.
4.  Copying the final result from device back to host (`cudaMemcpyDeviceToHost`).
5.  Freeing GPU memory (`cudaFree`).

```cpp
// In alexnet_cuda.cu (Illustrative)
void alexnetForwardPassCUDA(
    const std::vector<float>& input_host, /*...*/,
    std::vector<float>& output_host)
{
    // ... Calculate dimensions (Hc1, Wc1, Hp1, etc.) ...
    // ... Calculate buffer sizes (in_sz, c1_sz, p1_sz, etc.) ...

    float *d_input, *d_c1out, *d_p1out, *d_c2out, *d_p2out, *d_l2out;
    float *d_w1, *d_b1, *d_w2, *d_b2;

    // Allocate all device memory
    CUDA_CHECK(cudaMalloc(&d_input, in_sz * sizeof(float)));
    // ... Malloc for d_c1out, d_p1out, ..., d_w1, d_b1, etc. ...

    // Copy input data and parameters from host to device
    CUDA_CHECK(cudaMemcpy(d_input, input_host.data(), /*...*/, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, paramsConv1.weights.data(), /*...*/, cudaMemcpyHostToDevice));
    // ... Memcpy for biases, Conv2 weights/biases ...

    // --- Launch Kernels Sequentially ---
    // Block 1
    cudaConvLayer(d_c1out, d_input, d_w1, d_b1, H, W, C, K1, F1, S1, P1);
    cudaReluLayer(d_c1out, c1_sz); // c1_sz is total elements in d_c1out
    cudaMaxPoolLayer(d_p1out, d_c1out, Hc1, Wc1, C1, F_pool1, S_pool1);

    // Block 2 (d_p1out becomes input for Conv2)
    cudaConvLayer(d_c2out, d_p1out, d_w2, d_b2, Hp1, Wp1, C_conv2_input, K2, F2, S2, P2);
    cudaReluLayer(d_c2out, c2_sz);
    cudaMaxPoolLayer(d_p2out, d_c2out, Hc2, Wc2, C2, F_pool2, S_pool2);
    cudaLRNLayer(d_l2out, d_p2out, Hp2, Wp2, C2_out_lrn, N_lrn, alpha, beta, k_lrn);

    // Copy final result (d_l2out) from device to host
    output_host.resize(l2_sz);
    CUDA_CHECK(cudaMemcpy(output_host.data(), d_l2out, /*...*/, cudaMemcpyDeviceToHost));

    // Free all device memory
    CUDA_CHECK(cudaFree(d_input));
    // ... Free for d_c1out, d_p1out, ..., d_w1, d_b1, etc. ...
}
```

**CUDA Kernel Example - Convolution (`layers_cuda.cu`):**
A basic convolution kernel where each GPU thread computes one element of the output feature map.
```cuda
// In layers_cuda.cu
__global__ void convKernel(
    float* out, const float* in, const float* weights, const float* biases,
    int H, int W, int C_in,  // Input dimensions
    int K_out, int F, int S, int P, // Filter params
    int H_out, int W_out)           // Output dimensions
{
    int k = blockIdx.z * blockDim.z + threadIdx.z; // Output channel index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Output row index
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Output col index

    if (k >= K_out || y >= H_out || x >= W_out) return;

    float sum = biases[k];
    // Iterate over input channels, filter height, filter width
    for (int c = 0; c < C_in; ++c) {
        for (int fh = 0; fh < F; ++fh) {
            for (int fw = 0; fw < F; ++fw) {
                int input_y = y * S - P + fh;
                int input_x = x * S - P + fw;
                if (input_y >= 0 && input_y < H && input_x >= 0 && input_x < W) {
                    // Input index: (input_y, input_x, c)
                    // Weight index: (k, c, fh, fw)
                    sum += in[/*input_idx*/] * weights[/*weight_idx*/];
                }
            }
        }
    }
    // Output index: (y, x, k)
    out[/*output_idx*/] = sum;
}

// Kernel Launcher (Host-side)
void cudaConvLayer(/*...params...*/) {
    // Calculate H_out, W_out
    dim3 threadsPerBlock(16, 16, 1); // Example, adjust based on K_out
    dim3 numBlocks((W_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (H_out + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (K_out + threadsPerBlock.z - 1) / threadsPerBlock.z);
    convKernel<<<numBlocks, threadsPerBlock>>>(/*...args...*/);
    CUDA_CHECK(cudaGetLastError()); // Always check for kernel launch errors
}
```
*Note: The V3 CUDA kernels in the project were simpler 1D grid-stride loops, not 3D grids as shown above for illustration. The project's actual kernels map a 1D thread index to a 3D output coordinate.*

#### Key Files V3
-   `v3_cuda_only/src/main_cuda.cpp` (Host main)
-   `v3_cuda_only/src/alexnet_cuda.cu` (Host-side GPU orchestration)
-   `v3_cuda_only/src/layers_cuda.cu` (CUDA kernel implementations)
-   `v3_cuda_only/include/alexnet.hpp`, `layers.hpp`

#### Makefile Highlights V3
Uses `nvcc` for compiling both `.cu` (device/host) and `.cpp` (host-only) files. Specifies GPU architecture flags (e.g., `-gencode arch=compute_50,code=sm_50` for the Quadro M1200). The `--cudadevrt=none` flag was found crucial to resolve linker issues on some setups.

### V4: MPI+CUDA (Hybrid Host-Staging)

#### Goal and Paradigm V4
To combine MPI for inter-node/inter-process parallelism with CUDA for intra-node/intra-process GPU acceleration. This version uses a "host-staging" approach where data is explicitly moved between host memory (for MPI communication) and device memory (for CUDA computation).

#### Key Implementation Strategy & Code Examples V4
Each MPI rank is responsible for a slice of the input data and uses its assigned GPU.
1.  **Initialization:** MPI setup, each rank sets its GPU. Parameters broadcast.
2.  **Data Distribution:** Input data scattered row-wise to host buffers of each MPI rank.
3.  **Host-Side Halo Exchange:** For Conv1, MPI ranks exchange halo regions using their host buffers (`MPI_Isend/Irecv/Waitall`).
4.  **Padded Tile H$\rightarrow$D Copy:** Each rank copies its local data slice *plus the received halo regions* from its host buffer to its GPU's device memory.
5.  **GPU Tile Computation:** A helper function, `alexnetTileForwardCUDA` (in `alexnet_mpi_cuda.cu`), is called. This function:
    *   Takes the device pointer to the padded input tile.
    *   Internally allocates device memory for intermediate feature maps and parameters (weights/biases for the tile).
    *   Copies the global weights/biases (broadcast earlier to host) H$\rightarrow$D for the tile.
    *   Launches the full sequence of CUDA kernels (Conv1 $\rightarrow$ ReLU1 ... $\rightarrow$ LRN2) on the GPU, operating entirely on the data within that padded tile.
    *   Places the final output of LRN2 for that tile into a pre-allocated output device buffer.
    *   Frees its internal temporary device buffers.
6.  **Result Tile D$\rightarrow$H Copy:** The output tile from LRN2 is copied from device memory back to a host buffer.
7.  **Host-Side Trimming:** The host CPU then trims rows from this received tile that correspond to the initial halo padding, ensuring only the valid, owned output portion remains. This requires careful index calculation considering the transformations by all layers.
8.  **Result Aggregation:** The trimmed, valid local output slices are gathered by rank 0 using `MPI_Gatherv`.

**Host-Side Orchestration (`main_mpi_cuda.cpp`):**
```cpp
// In main_mpi_cuda.cpp
// ... MPI_Init, rank, size, GPU setup (cudaSetDevice) ...
// ... Broadcast parameters (p1, p2 globally) ...
// ... Scatter input rows to `myIn` (std::vector<float> on host) ...

// Host Halo Exchange for Conv1 (Conceptual)
// `myIn` is updated/expanded with halos from neighbors (similar to V2.2 logic)

// Device Copy H->D
float *d_in_padded_tile = nullptr;
CUDA_CHECK(cudaMalloc(&d_in_padded_tile, myIn.size() * sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_in_padded_tile, myIn.data(), myIn.size() * sizeof(float), cudaMemcpyHostToDevice));

// Calculate output dimensions for the *padded tile* (Hp2, Wp2, p2.K)
// These are the dimensions of the output if the entire padded tile was processed.
std::vector<float> host_tile_output_buffer( (size_t)Hp2_padded * Wp2_padded * p2.K );
float* d_out_padded_tile = nullptr;
CUDA_CHECK(cudaMalloc(&d_out_padded_tile, host_tile_output_buffer.size() * sizeof(float)));

// GPU Tile Computation
// `paddedH_for_gpu` is the height of `myIn` (local rows + halo rows)
alexnetTileForwardCUDA(d_in_padded_tile, p1, p2, paddedH_for_gpu, W_original, C_original, d_out_padded_tile);

// Device Copy D->H
CUDA_CHECK(cudaMemcpy(host_tile_output_buffer.data(), d_out_padded_tile, /*...*/, cudaMemcpyDeviceToHost));

// Host Trimming
// `start_row_idx`, `num_valid_rows` are calculated based on rank, halo propagation through all layers.
// `localFinalOutput` (std::vector<float>) gets the valid portion from `host_tile_output_buffer`.

// Gather `localFinalOutput`
// ... MPI_Gatherv ...
// ... cudaFree(d_in_padded_tile), cudaFree(d_out_padded_tile) ...
// ... MPI_Finalize ...
```

**GPU Tile Computation (`alexnet_mpi_cuda.cu`):**
```cuda
// In alexnet_mpi_cuda.cu
void alexnetTileForwardCUDA(const float* d_input_tile_padded, // Padded input for this rank's slice
                            const LayerParams& p1_global, const LayerParams& p2_global,
                            int H_tile_padded, int W_original, int C_original, // Dimensions of d_input_tile_padded
                            float* d_output_tile_final) // Pre-allocated output buffer for the tile
{
    // Calculate intermediate dimensions based on H_tile_padded, W_original, C_original
    // ... Hc1, Wc1, Hp1, Wp1, C1_out ...
    // ... Hc2, Wc2, Hp2, Wp2, C2_out ...

    // Allocate temporary device buffers for intermediate layers (d_c1, d_p1, d_c2, d_p2)
    // Allocate device buffers for weights/biases (dw1, db1, dw2, db2)
    // Copy p1_global.weights, p1_global.biases, etc. to dw1, db1 etc. (H->D)

    // Execute kernel sequence (similar to V3's alexnetForwardPassCUDA but on tile data)
    cudaConvLayer(d_c1, d_input_tile_padded, dw1, db1, /*...*/);
    cudaReluLayer(d_c1, /*...*/);
    cudaMaxPoolLayer(d_p1, d_c1, /*...*/);
    // ... Conv2, ReLU2, Pool2 ...
    cudaLRNLayer(d_output_tile_final, d_p2, /*...*/); // Final result into d_output_tile_final

    // Free temporary device buffers (dw1, db1, d_c1, d_p1, etc.)
}
```
This host-staging approach simplifies some logic by having the GPU process a self-contained tile but introduces significant H$\leftrightarrow$D data movement for the full tile (including halos) and for parameters within `alexnetTileForwardCUDA` on each call, becoming a major bottleneck.

#### Key Files V4
-   `v4_mpi_cuda/src/main_mpi_cuda.cpp`
-   `v4_mpi_cuda/src/alexnet_mpi_cuda.cu`
-   `v4_mpi_cuda/src/layers_mpi_cuda.cu` (CUDA kernels, similar to V3)
-   `v4_mpi_cuda/include/alexnet.hpp`, `layers.hpp`

#### Makefile Highlights V4
Uses `nvcc` as the main compiler, configured to use `mpicxx` as the host C++ compiler via the `-ccbin=mpicxx` flag. This allows `nvcc` to correctly handle both CUDA device code and host code that includes MPI headers and calls MPI functions. Similar GPU architecture flags and `--cudadevrt=none` are used.

---

## Compilation and Execution Summary

-   **Compilation:** Navigate to the specific version directory (e.g., `final_project/v1_serial/`) and run `make clean && make`. This will produce an executable named `template`.
-   **Execution:**
    -   **V1 (Serial) & V3 (CUDA Only):** `./template`
    -   **V2.1, V2.2 (MPI Only) & V4 (MPI+CUDA):** `mpirun -np <N> ./template`
        -   Replace `<N>` with the desired number of MPI processes (e.g., 1, 2, 4).
        -   For V4, each MPI process will attempt to use a GPU.
        -   On a cluster, a hostfile (e.g., `mpirun -np 4 -hostfile myhosts ./template`) might be necessary.
        -   If running locally with `<N>` greater than physical CPU cores, the `--oversubscribe` flag for `mpirun` might be needed.

---

## Summary of Performance Findings and Conclusions

The project systematically evaluated four distinct parallelization strategies for AlexNet inference Blocks 1 & 2. Median-based runtimes were used for robust comparisons.

### Performance Highlights
*(Based on median runtimes from the project_final_scorecard_median_recalc_cv.md file)*
-   **V1 (Serial CPU):** Established a baseline median runtime of **\SI{0.784}{s}** for NP=1.
-   **V2.1 (MPI Broadcast):** Demonstrated poor scalability, with median runtime increasing from \SI{0.743}{s} (NP=1) to \SI{0.819}{s} (NP=4). The broadcast overhead was significant.
-   **V2.2 (MPI Scatter+Halo):** Showed effective CPU-side parallelism. Median runtime improved from \SI{0.714}{s} (NP=1) to \SI{0.287}{s} (NP=4), achieving a median-based speedup of **2.49x** relative to its own NP=1 performance.
-   **V3 (CUDA Only):** Provided the best single-instance (NP=1) performance with a median runtime of **\SI{0.488}{s}**, a 1.61x speedup over V1.
-   **V4 (MPI+CUDA Host-Staging):**
    -   Achieved a median NP=1 runtime of **\SI{0.429}{s}**, slightly faster than V3 NP=1, likely due to minor differences or measurement variance, but indicating the MPI overhead for a single process was manageable.
    -   However, it scaled very poorly. Median runtime increased to \SI{0.401}{s} at NP=4, yielding a median-based speedup of only **1.07x** relative to its own NP=1. This was slower than V2.2 Scatter+Halo at NP=4.
    -   The Coefficient of Variation (CV) for V4 at NP=1 was notably high (0.773), indicating significant runtime instability compared to other versions at NP=1 (e.g., V1 CV: 0.214; V3 CV: 0.353).

### Key Bottlenecks Identified
-   **V2.1 (MPI Broadcast):** The broadcast operation itself.
-   **V3 (CUDA Only):** While fast, performance is limited by PCIe H$\leftrightarrow$D transfers for the entire dataset and parameters, and the efficiency of the basic CUDA kernels.
-   **V4 (MPI+CUDA Host-Staging):** This version was severely limited by:
    1.  **Host-Staging Data Path:** MPI communication happening on host buffers, requiring explicit `cudaMemcpy` calls to move data (local slice + received halos) to the GPU, and results back to the host. These PCIe transfers are significant overheads.
    2.  **Synchronization:** Implicit and explicit synchronization points between MPI calls, CUDA H$\leftrightarrow$D copies, and kernel launches can serialize parts of the execution.
    3.  **CPU-Side Logic:** Trimming halo regions on the CPU after D$\leftrightarrow$H copy adds to host processing time.
    4.  **Small Compute per GPU:** As NP increases, the data slice per GPU shrinks. For this problem size and implementation, the overhead of data movement and MPI coordination outweighed the benefits of distributing the (already fast on one GPU) V3 computation. GPUs likely experienced significant idle time.

### Overall Lessons Learned
1.  **Parallelization Paradigm Impact:** The choice of paradigm (MPI, CUDA, Hybrid) and its specific implementation strategy (e.g., broadcast vs. scatter+halo, host-staging vs. device-direct) drastically affects performance and scalability.
2.  **Data Movement is Critical:** Minimizing data movement (across network for MPI, across PCIe for CUDA H$\leftrightarrow$D) is paramount in parallel systems. V4's performance highlighted this.
3.  **Scalability Isn't Automatic:** Adding more resources (CPUs/GPUs) doesn't guarantee better performance if communication or other overheads become dominant (Amdahl's Law).
4.  **Hybrid Complexity:** MPI+CUDA models are powerful but significantly more complex to implement, debug, and optimize correctly. Synchronization and data flow between host and device across multiple ranks require meticulous design.
5.  **Profiling is Essential:** Performance results often indicate *what* is happening, but deep profiling (e.g., with Nsight Systems/Compute) is necessary to understand *why* and to guide targeted optimizations.
6.  **Environment and Tooling Matter:** Linker issues and compiler flags (like `--cudadevrt=none`) underscore the importance of understanding the development environment.
7.  **Problem Size Influence:** The performance characteristics observed are for a single image inference. Larger batch sizes might amortize some overheads differently.

---

## Future Work Highlights
Based on the project findings, key areas for future work include:
1.  **V5: CUDA-Aware MPI:** Implement V5 to allow MPI calls to operate directly on GPU device pointers. This is the most direct way to address V4's host-staging bottleneck by potentially eliminating H$\leftrightarrow$D copies for halo exchanges.
2.  **Detailed Profiling:** Use NVIDIA Nsight Systems and Compute to thoroughly profile V3 and V4 (and a future V5) on the target cluster to pinpoint exact locations of bottlenecks (e.g., H$\leftrightarrow$D transfer times, kernel execution times, MPI wait times).
3.  **Asynchronous Operations:** Explore asynchronous techniques (CUDA streams for compute/copy overlap, non-blocking MPI calls, pinned host memory) to improve V4/V5 performance by hiding latencies.
4.  **Kernel Optimizations:** Refine CUDA kernels in V3/V4/V5 using techniques like shared memory tiling for data reuse, ensuring coalesced global memory access, and optimizing for warp efficiency.
5.  **Investigate Numerical Differences:** Conduct a more thorough layer-by-layer comparison between CPU and GPU outputs to pinpoint sources of minor numerical variations.
