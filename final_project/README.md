# CS485: GPU Cluster Programming (MPI+CUDA) – Final Project

This repository holds the **final project** for the CS485 course at NJIT (taught by Andrew Sohn), focusing on **AlexNet inference** over multiple CUDA-capable nodes using **MPI + CUDA**. Our primary goal is to demonstrate data-parallel CNN inference on a cluster of at least two GPUs, showing measurable speedups over single-GPU and single-process approaches.

---

## Table of Contents
1. [Project Objectives](#project-objectives)
2. [High-Level Design](#high-level-design)
3. [Repository Layout](#repository-layout)
4. [Architecture & Implementation](#architecture--implementation)
5. [Code Snippets & Highlights](#code-snippets--highlights)
6. [Build & Run Instructions](#build--run-instructions)
7. [Performance & Benchmarks](#performance--benchmarks)
8. [Testing & Automation](#testing--automation)
9. [Future Extensions](#future-extensions)
10. [Troubleshooting](#troubleshooting)
11. [References & Resources](#references--resources)

---

## 1. Project Objectives

1. **Implement Data-Parallel AlexNet Inference**: Each MPI rank has a full copy of the AlexNet weights, processes its subset of the input batch, and optionally gathers results.
2. **Use CUDA for GPU Acceleration**: Maximize concurrency by offloading convolution, pooling, and LRN layers onto the GPU. Minimize memory transfers using pinned host memory and efficient device buffers.
3. **Leverage MPI for Multi-Node Execution**: Synchronize or broadcast model weights (and input data if needed) from the master rank to worker ranks. Potentially gather inference results for final logging.
4. **Automate Workflows**: Provide scripts to build, run, and collect performance logs. Ensure each run captures meaningful timing for kernel execution and MPI overhead.
5. **Demonstrate Speedups**: Show that multiple GPUs working in parallel can handle significantly larger input sets faster than single-GPU or single-process approaches.

---

## 2. High-Level Design

### Approach: Data Parallelism
- Each GPU rank gets a portion of the input (e.g., a fraction of images in a dataset).  
- The rank copies data to device memory and performs AlexNet’s forward pass.  
- Final inference outputs (class probabilities) are gathered on rank 0 using `MPI_Gather`.

### AlexNet Blocks Implemented
1. **Block1**: Conv1 → ReLU1 → MaxPool1  
2. **Block2**: Conv2 → ReLU2 → MaxPool2 → LRN2  

The final output shape after Block2 is 256×13×13 = 43,264 floats for batch size 1. (We can expand for multiple images in future.)

### Potential Next Blocks (Future):
- **Conv3, Conv4, Conv5**  
- **Fully-Connected Layers (FC6, FC7, FC8)**  
- **Softmax Classification**  

---

## 3. Repository Layout

```bash
final_project/
├── include/
│   ├── alexnet.hpp      # Public interface for alexnetForward()
│   ├── layers.hpp       # Declarations for convolution, pooling, LRN kernels
│   └── mpi_helper.hpp   # MPI utility functions (if needed)
├── src/
│   ├── main.cpp         # Entry point; initializes MPI, reads input, runs alexnetForward
│   ├── alexnet_hybrid.cu# Primary pipeline for AlexNet forward pass (Block1 & Block2)
│   ├── layers.cu        # CUDA kernels (conv, relu, pool, lrn) + host launchers
│   └── conv_test.cpp    # A standalone test to validate convolution kernels
├── Makefile             # Builds the 'template' executable & conv_test
└── scripts/
    ├── run_final_project.sh     # Automates build & run
    ├── test_final.sh            # Another example test script
    └── summarize_results.sh     # Aggregates logs & prints summary
```

Key points:
- **`alexnet_hybrid.cu`** orchestrates memory allocation, kernel launches for each block, and final data copy-out.
- **`layers.cu`** implements naive convolution, pooling, and local response normalization kernels, each with their own launchers.
- **`main.cpp`** handles MPI initialization, broadcasts weights, divides input among ranks, and calls `alexnetForward`.

---

## 4. Architecture & Implementation

### Data Parallelism
- **MPI_Bcast** to distribute model weights from rank 0 to all ranks.
- Each rank receives a distinct subset of input images or a multiplier-based synthetic data set.
- After computing forward pass, we gather outputs to rank 0 with **MPI_Gather**.

### Naive CUDA Kernels
1. **Convolution** (Conv Layer):
   - 3D grid: \[ (W_out,H_out), K \] – each block corresponds to an output pixel region and one filter.
   - Each thread computes a single output element by iterating through channels & kernel elements.  
2. **ReLU**:
   - 1D kernel that simply applies `max(0, x)` in place.
3. **Pooling** (MaxPool):
   - Another 3D grid: \[ (W_out,H_out), Channels \], each thread scans a `poolSize × poolSize` region in the input.  
4. **LRN** (Local Response Normalization):
   - Cross-channel naive approach: sum squares of neighbor channels, scale the value. Typical constants: alpha=1e-4, beta=0.75, k=2.

### Memory Flows
- **Device Buffers**: 
  - `d_input` → `d_pool1_out` → `d_conv2_out` → `d_pool2_out` → `d_lrn2_out`
- **Weights**: Copied once from host to device (if we were truly using them within each block).
- **Output**: Copied back to host from `d_lrn2_out` at the end.

### Debugging & Performance
- We use `cudaEvent_t` to measure kernel execution times. 
- MPI timers (`MPI_Wtime`) measure the overall forward pass per rank.  
- All logs get appended to a user-defined log folder for summary.

---

## 5. Code Snippets & Highlights

### 5.1 `alexnet_hybrid.cu` Excerpt

```cpp
// Pseudocode snippet from block2
// ...
// 1) Conv2
ConvLayerParams conv2_params = {
    /* inputChannels */ 96,
    /* outputChannels */ 256,
    /* kernelSize */ 5,
    /* stride */ 1,
    /* padding */ 2,
    /* inputHeight */ 27,
    /* inputWidth */ 27,
    /* outputHeight */ 27,
    /* outputWidth */ 27
};
launch_conv2d_forward_conv2(d_pool1_out, d_conv2_out, d_conv2_weights, d_conv2_biases, conv2_params);

// 2) ReLU2 (in-place)
int conv2_num_elems = 256 * 27 * 27;
launch_relu_forward_conv2(d_conv2_out, conv2_num_elems);

// 3) MaxPool2
PoolLayerParams pool2_params = {
    3, 2, 256, 27, 27, 13, 13
};
launch_maxpool_forward2(d_conv2_out, d_pool2_out, pool2_params);

// 4) LRN2
LRNLayerParams lrn2_params = {
    256, 13, 13, 5, 1.0e-4f, 0.75f, 2.0f
};
launch_lrn_forward(d_pool2_out, d_lrn2_out, lrn2_params);
// ...
```

### 5.2 `layers.cu` Snippet

```cpp
// Local Response Normalization kernel
__global__ void lrn_forward_kernel(
    const float* input, float* output, LRNLayerParams params)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if (c < params.channels && y < params.height && x < params.width) {
        int idx = c * (params.height * params.width) + y * params.width + x;
        int halfWin = (params.localSize - 1) / 2;
        float accum = 0.0f;
        // Summation over neighboring channels
        for (int c2 = max(0, c-halfWin); c2 <= min(params.channels-1, c+halfWin); c2++) {
            int idx2 = c2 * (params.height * params.width) + y * params.width + x;
            float val = input[idx2];
            accum += val * val;
        }
        float denom = powf(params.k + params.alpha * accum, params.beta);
        output[idx] = input[idx] / denom;
    }
}
```

---

## 6. Build & Run Instructions

1. **Go to Final Project Directory:**
   ```bash
   cd final_project
   ```
2. **Clean & Build:**
   ```bash
   make clean
   make
   ```
   This produces:
   - `template`: Main AlexNet inference code (MPI+CUDA).
   - `conv_test`: Optional unit test for convolution.

3. **Run Locally with MPI:**  
   ```bash
   mpirun --oversubscribe -np 2 ./template
   ```
   - By default, we generate synthetic input data. The rank 0 can also broadcast real weights, if provided.

4. **Log & Summaries:**  
   - If using scripts, do:
     ```bash
     bash scripts/run_final_project.sh
     bash scripts/summarize_results.sh
     ```
   - These scripts build the project, run multiple test configurations, and generate a summary of forward pass times and kernel timings.

**Reminder:** For final grading, we must ensure everything runs under **Fedora 37**, with **GCC 12** and **CUDA 12.x**.

---

## 7. Performance & Benchmarks

### Local Tests
- For small synthetic inputs (batch=1):
  - We see each block (Conv1+Pool1, Conv2+Pool2+LRN2) taking under 1 second per rank for typical 227×227 input.
  - Kernel-level timings are in the range of milliseconds.

### Scaling with Multiple Ranks
- With 2 ranks (2 GPUs), each rank processes half the images. We expect near 2× speedup on a multi-node cluster, subject to overhead from MPI_Bcast and MPI_Gather.
- We capture per-rank times using `MPI_Wtime()` and gather them for summary.

### Example Observed Results
```
Process 0 forward pass time (MPI_Wtime): 0.402123 seconds
Process 1 forward pass time (MPI_Wtime): 0.401892 seconds
Kernel execution time: Conv2->LRN2 block: ~1.2 ms
...
```

---

## 8. Testing & Automation

**Test Scripts**:

- **`scripts/run_final_project.sh`**: 
  - Builds the final project and runs with multiple ranks (1,2,4).
  - Times each run, logs outputs, checks for potential timeouts (>30s).
- **`scripts/summarize_results.sh`**: 
  - Parses logs for lines like `Kernel execution time:` or `forward pass time (MPI_Wtime):`.
  - Aggregates them into a final table for quick referencing.

**Unit Tests**:

- **`conv_test`**: 
  - Exercises just the convolution kernel on synthetic input (like 3×227×227 → 96×27×27).
  - Verifies basic correctness and prints partial results.

---

## 9. Future Extensions

1. **Additional AlexNet Layers**:  
   - **Conv3–Conv5** with ReLU and pooling as needed.  
   - **Fully Connected Layers (FC6, FC7, FC8)** for classification.  
2. **Training Mode**:  
   - Implement backward pass for each layer.  
   - Use `MPI_Allreduce` to aggregate gradients across ranks in data-parallel training.  
3. **Model Parallelism**:  
   - Partition filters across GPUs for extremely large models or memory constraints.  
4. **Performance Optimization**:  
   - Use shared memory tiling in convolution.  
   - Explore library calls (cuBLAS, cuDNN) for high-performance GEMM and conv.  
   - Overlap communication with computation using CUDA streams + nonblocking MPI calls.

---

## 10. Troubleshooting

1. **Kernel Launch Failures**:  
   - Check grid/block dimensions. If output dims exceed your launch parameters, indexes might be out of range.
2. **MPI Hang or Timeout**:  
   - Possibly mismatched MPI calls (e.g., one rank calls Bcast, another calls Scatter).  
   - Use `scripts/check_cluster.sh` to confirm node connectivity.  
3. **Makefile Errors**:  
   - Ensure that `layers.cu` is included in sources so definitions link properly.  
   - Verify real **TAB** characters in recipes (No spaces).
4. **Mismatch in Output**:  
   - Confirm consistent initialization of weights, bias, or random data across ranks. If ranks see different data, results differ.

---

## 11. References & Resources

- **MPI Standard (v3.1)**: [mpi-forum.org](https://www.mpi-forum.org/)  
- **NVIDIA CUDA Docs**: [docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)  
- **AlexNet Paper**: Krizhevsky, Sutskever, Hinton (NIPS 2012)  
- **Programming Massively Parallel Processors (4th Ed.)**: Kirk, Hwu, El Hajj  
- **Official HPC Channels**: Slack, NVIDIA Developer Forums, HPC mailing lists.
