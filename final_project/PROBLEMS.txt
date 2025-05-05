# CS485 Final Project: Problem Log & Tracking (Enhanced)

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation - Blocks 1&2)
**Version:** 1.1 (Deep Research Update)
**Purpose:** This document serves as a comprehensive, research-informed log of problems, issues, errors, warnings, and challenges encountered (past), currently being addressed (present), and anticipated (future) during the development of the CS485 final project. It aims to provide deep context on the nature of these problems and potential solutions.

---

## Table of Contents

1.  [Past Problems & Resolutions](#1-past-problems--resolutions)
    *   [V1 Serial](#v1-serial)
    *   [V2 MPI Only](#v2-mpi-only)
    *   [V3 CUDA Only](#v3-cuda-only)
    *   [General Development](#general-development-1)
2.  [Present Problems & Current Debugging Focus](#2-present-problems--current-debugging-focus)
    *   [V4 MPI+CUDA Hybrid: Runtime Crash (NP=4)](#v4-mpicuda-hybrid-runtime-crash-np4)
    *   [V4 MPI+CUDA Hybrid: Output Format Mismatch](#v4-mpicuda-hybrid-output-format-mismatch)
    *   [V4 MPI+CUDA Hybrid: Correctness Verification (Numerical & Trimming)](#v4-mpicuda-hybrid-correctness-verification-numerical--trimming)
    *   [V4 MPI+CUDA Hybrid: Potential `alexnetTileForwardCUDA` Inefficiencies](#v4-mpicuda-hybrid-potential-alexnettileforwardcuda-inefficiencies)
    *   [V3 CUDA Only: Performance Bottleneck](#v3-cuda-only-performance-bottleneck)
    *   [Cross-Version Numerical Discrepancies](#cross-version-numerical-discrepancies-1)
3.  [Anticipated Future Problems & Challenges](#3-anticipated-future-problems--challenges)
    *   [V4 Optimization: Asynchronous Overlap Implementation](#v4-optimization-asynchronous-overlap-implementation)
    *   [V4 Optimization: Host-Side Trimming Logic](#v4-optimization-host-side-trimming-logic)
    *   [V5 CUDA-Aware MPI: Configuration & Performance Validation](#v5-cuda-aware-mpi-configuration--performance-validation)
    *   [Advanced CUDA Kernel Optimization](#advanced-cuda-kernel-optimization)
    *   [Scalability Limits & Amdahl's Law](#scalability-limits--amdahls-law)
    *   [Target Cluster Environment Variations](#target-cluster-environment-variations)
    *   [Full AlexNet Extension (Optional)](#full-alexnet-extension-optional-1)

---

## 1. Past Problems & Resolutions

This section details issues identified and resolved during earlier stages, providing context on the development process.

### V1 Serial
*   **Issue:** Minor logical errors in initial layer implementations (e.g., off-by-one errors in loop bounds for convolution/pooling, incorrect indexing math `idx3D`, LRN formula errors).
    *   **Resolution:** Standard debugging practices: using `g++ -g`, stepping through with GDB, adding print statements for intermediate dimensions and values, comparing results against hand calculations or reference implementations for small test cases. Focused on achieving functional correctness layer-by-layer.
*   **Issue:** Correct calculation of output feature map dimensions after Conv/Pool layers, considering padding and stride.
    *   **Resolution:** Implemented and tested standalone helper functions (`calculate_conv_output_dims`, `calculate_pool_output_dims`) based on standard CNN formulas, ensuring consistent usage.

### V2 MPI Only
*   **Issue (V2.1):** Severe performance degradation with increasing processes.
    *   **Resolution:** Confirmed as expected behavior for the naive "broadcast all, compute all, gather slice" strategy due to O(P) communication overhead and lack of data distribution. Retained as a pedagogical baseline comparison.
*   **Issue (V2.2):** Implementing non-blocking MPI halo exchange (`MPI_Isend`/`Irecv`/`Waitall`) correctly without deadlocks or race conditions.
    *   **Resolution:** Adhered to standard non-blocking patterns: posting all receives (`Irecv`), then posting all sends (`Isend`), then waiting (`Waitall`) on *all* requests. Used distinct MPI tags for up/down communication to prevent message mismatch. Tested incrementally (NP=2 first). Ensured send buffers were not reused before `Waitall` completed.
*   **Issue (V2.2):** Implementing the complex logic for asymmetric trimming of intermediate feature maps after pooling layers. This is required because halo data propagates through layers, affecting more output rows than the initial halo size.
    *   **Resolution:** Derived formulas to calculate the number of "invalid" rows at the top and bottom of each rank's local output based on the initial halo size (`pad`), layer strides (`S`, `S_pool`), and filter sizes. Implemented careful indexing and buffer copies (`std::copy_n`) to extract the valid data region before the final gather. This logic (in `main.cpp`) was likely debugged iteratively.
*   **Issue (V2.2):** Correct computation of counts/displacements for `MPI_Scatterv`/`MPI_Gatherv` with non-uniform data distribution (`H % size != 0`).
    *   **Resolution:** Rank 0 calculates arrays (`sendCnt`, `sendDisp`, `recvCnt`, `recvDisp`) based on integer division (`base = H/size`) and remainder (`rem = H%size`), ensuring all rows/elements are accounted for. These calculation arrays were then used in the collective calls.

### V3 CUDA Only
*   **Issue:** CUDA kernel correctness: Potential for out-of-bounds memory access due to incorrect index calculations within kernels, race conditions (if using shared memory improperly - not apparent in current basic kernels), incorrect handling of boundary conditions.
    *   **Resolution:** Use of `cuda-memcheck` during testing to detect illegal memory accesses. Careful review of kernel indexing logic (global thread ID `idx` to multi-dimensional `h,w,c` mapping). Use of `CUDA_CHECK` macro after kernel launches (`cudaGetLastError()`) to catch launch configuration errors or asynchronous errors from previous kernels.
*   **Issue:** CUDA API usage errors: Mismatched `cudaMemcpy` sizes, incorrect `cudaMemcpyKind` (H->D, D->H, D->D), forgetting to allocate/free device memory, incorrect kernel launch parameters (grid/block dimensions).
    *   **Resolution:** Consistent use of the `CUDA_CHECK` macro around *all* CUDA runtime API calls. Double-checking calculated buffer sizes against `sizeof(float)` and dimensions. Ensuring `cudaFree` is called for all `cudaMalloc`'d pointers. Calculating grid/block dimensions to cover the entire problem space (`grid = (N + blockDim - 1) / blockDim`).

### General Development
*   **Issue:** Managing build dependencies and flags across C++, MPI, and CUDA using Makefiles. Ensuring correct compilation (`-dc` for device linking in V4) and linking order (`-lcudart`, MPI flags from wrapper).
    *   **Resolution:** Evolved Makefiles for each version. V4 Makefile leverages `nvcc -ccbin=mpicxx`, `mpicxx --showme` for MPI flags, and uses standard CUDA compilation practices (`-gencode`, separate host/device compilation if needed - V4 uses `-dc`).
*   **Issue:** Header file management (`#include` paths, header guards `#pragma once` or `#ifndef/#define`).
    *   **Resolution:** Consistent use of relative paths (`../include/`) and header guards.

---

## 2. Present Problems & Current Debugging Focus

This section provides deeper analysis and diagnostic approaches for active issues.

### V4 MPI+CUDA Hybrid: Runtime Crash (NP=4)
*   **Problem:** V4 fails with MPI Exit Code 134 (often corresponds to SIGABRT - Abort signal) when run with 4 processes. Runs complete for NP=1, 2.
*   **Status:** Confirmed via `run_final_project.sh`. Log file analysis and debugging required.
*   **Impact:** Prevents validation and analysis at NP=4. Indicates a scalability bug or resource issue.
*   **Research-Informed Potential Causes:**
    *   **MPI Communication Error:**
        *   *Incorrect Counts/Displacements:* Calculation of `sendCnt`/`recvCnt` or `sendDisp`/`recvDisp` for `MPI_Scatterv`/`MPI_Gatherv` or sizes for `MPI_Isend`/`Irecv` might be incorrect specifically for the NP=4 partitioning, leading to buffer overflows/underflows or access violations triggering an abort.
        *   *Deadlock:* While non-blocking calls are used, an error in the `Waitall` logic or an unexpected blocking condition in a specific rank's path at NP=4 could lead to a deadlock, which MPI might eventually abort.
        *   *Resource Limits:* Exceeding MPI message buffer limits (system-defined or Open MPI parameters like `btl_tcp_eager_limit`) might occur with the communication pattern at NP=4.
    *   **Memory Errors:**
        *   *Host Memory:* An incorrect calculation related to NP=4 might lead to out-of-bounds access in host vectors (`myIn`, `tileOut`, `local`) during padding, copying, or trimming.
        *   *Device Memory:* Similar incorrect calculations could lead to out-of-bounds access within CUDA kernels or `cudaMemcpy` operations, potentially only triggered by the specific data dimensions/access patterns of certain ranks at NP=4. GPU memory exhaustion is also possible if allocation sizes scale incorrectly with NP (though less likely in this data-parallel approach).
    *   **CUDA Errors:** A CUDA kernel or API call might be failing only on a specific rank at NP=4 (e.g., due to invalid launch parameters derived from NP=4 dimensions), and the `CUDA_CHECK` macro correctly calls `MPI_Abort`.
    *   **Environment/Resource Limits:** Exceeding per-process memory limits, limits on open file descriptors/sockets used by MPI, or specific WSL2 resource constraints when running 4 processes concurrently.
*   **Recommended Diagnostic Steps:**
    1.  **Analyze Log:** Thoroughly examine `logs/final_project_v4_np4.log` for any specific error messages from MPI, CUDA, or stderr.
    2.  **MPI Debug Flags:** Run with increased Open MPI verbosity or debug flags (e.g., `mpirun --mca btl_base_verbose 100 ...` or `mpirun --debug-daemons ...`).
    3.  **Reproduce Manually & Isolate:** Run `mpirun -np 4 ./template` directly. Add detailed rank-based `fprintf(stderr, ...)` statements before/after MPI calls, memory operations, H<->D copies, kernel launches to pinpoint the failure location.
    4.  **Debuggers:** Use `gdb --args mpirun -np 4 ./template` (requires debug symbols `-g`). Set breakpoints before suspect calls. For hangs, attach `gdb` to running processes. Use `cuda-gdb` similarly for device-side debugging.
    5.  **Memory Checkers:** Run `cuda-memcheck mpirun -np 4 ./template`. This is highly effective for finding CUDA memory errors (invalid access, misaligned, etc.) even if they don't always cause crashes. Check host memory using Valgrind (though complex with MPI: `valgrind --tool=memcheck mpirun ...` might work with specific MPI setups).
    6.  **Resource Monitoring:** Use `top`/`htop` or node-level monitoring to check memory/CPU usage during the NP=4 run. Check `dmesg` for system-level errors.
    7.  **Simplify:** Temporarily comment out sections (e.g., second halo exchange, specific kernels) to isolate the failing component.

### V4 MPI+CUDA Hybrid: Output Format Mismatch
*   **Problem:** `cout` labels in `main_mpi_cuda.cpp` (`shape`, `sample`) differ from script parser expectations (`Final Output Shape:`, `Final Output (first 10 values):`).
*   **Status:** Identified. Simple fix required.
*   **Impact:** Prevents automated result parsing and performance logging by `run_final_project.sh`.
*   **Resolution:** Modify the `std::cout` lines in the rank 0 block of `main_mpi_cuda.cpp` to precisely match the required labels. Add time output consistent with other versions.

### V4 MPI+CUDA Hybrid: Correctness Verification (Numerical & Trimming)
*   **Problem:** Need to rigorously verify V4 numerical output against V1/V3 and ensure host-side trimming logic is correct.
*   **Status:** Pending fix for crash/output format.
*   **Impact:** Essential for valid performance comparison and confirming successful hybrid implementation.
*   **Research-Informed Challenges & Verification Steps:**
    *   *Trim Logic Complexity:* Asymmetric trimming based on halo propagation through multiple layers (Conv/Pool with different strides/pads) is non-trivial. The current V4 `start`/`stop` logic seems simpler than V2.2's derived `trim_top1/trim_bot1` etc., and needs verification. Off-by-one errors in calculating valid row ranges are common.
    *   *Numerical Stability:* Parallel execution order can affect floating-point results slightly (non-associativity). Need to compare using a tolerance (epsilon). Large deviations likely indicate bugs.
    *   *Verification Strategy:*
        1.  Use fixed input data and fixed initial weights/biases (not random) for consistent runs.
        2.  Compare V4 output (element-wise) against V1/V3 output using a small epsilon (e.g., 1e-5).
        3.  Add detailed rank-based prints *before* the gather in V4 showing `localH` (input rows), `paddedH` (after halo), calculated intermediate dimensions (Hc1, Hp1, Hc2, Hp2), calculated final `local` output height, and the calculated `start`/`stop` trim indices. Compare these dimensions across ranks and against manual calculations.
        4.  Visualize intermediate results (e.g., write small feature maps to files) if discrepancies are hard to track.

### V4 MPI+CUDA Hybrid: Potential `alexnetTileForwardCUDA` Inefficiencies
*   **Problem:** The `alexnetTileForwardCUDA` helper function likely re-allocates/copies parameters (weights/biases) H->D on *every call* (once per rank), which is inefficient.
*   **Status:** Suspected based on code structure where `main` passes host `LayerParams` to the tile function.
*   **Impact:** Adds significant, redundant H->D transfer overhead, negatively impacting V4 performance.
*   **Verification/Resolution:** Examine `alexnetTileForwardCUDA` code. If confirmed, refactor V4: allocate/copy parameters (`d_weights1`, `d_biases1`, etc.) *once* per rank after MPI setup, then pass these *device pointers* to `alexnetTileForwardCUDA` (or a modified version) which then directly uses them for kernel launches.

### V3 CUDA Only: Performance Bottleneck
*   **Problem:** V3 performance is much worse than V1 (Serial).
*   **Status:** Observed. Profiling required.
*   **Impact:** Defeats purpose of GPU acceleration. Highlights implementation inefficiency.
*   **Research-Informed Potential Causes:**
    *   **Memory Transfer Overhead:** Copying large input, weights, biases, and output between host (pageable `std::vector`) and device using synchronous `cudaMemcpy` is often the dominant factor for non-optimized CUDA code.
    *   **Kernel Inefficiency:**
        *   *Memory Access Patterns:* Non-coalesced global memory access in kernels (especially Conv) significantly degrades bandwidth. Lack of shared memory tiling to exploit data reuse in Conv.
        *   *Low Occupancy:* Insufficient active warps per SM due to high register usage, high shared memory usage, or insufficient parallelism (block count/size). Check using Nsight Compute or occupancy calculator.
        *   *Instruction Latency/Throughput:* Kernels might be bound by arithmetic instruction latency or low throughput, although memory is more common for naive Conv kernels.
    *   **Kernel Launch Latency:** Overhead associated with launching many small kernels (though fewer here as layers are somewhat large).
*   **Recommended Diagnostic Steps:**
    1.  **Nsight Systems (`nsys profile ./template`):** Visualize the application timeline. Measure time spent in H->D memcpy, D->H memcpy, and individual kernel executions. Identify the largest time components.
    2.  **Nsight Compute (`ncu --set full -o profile_v3 ./template`):** Profile individual kernels (especially `convKernel`). Analyze memory throughput (L1/L2 cache hits, DRAM bandwidth), compute throughput, occupancy, instruction stalls, and source-level performance metrics to pinpoint specific inefficiencies. Check for coalescing issues, shared memory conflicts (if used), register spilling.
    3.  **Experiment:** Test impact of using pinned host memory (`cudaMallocHost`/`cudaFreeHost`) for input/output buffers instead of `std::vector`.

### Cross-Version Numerical Discrepancies
*   **Problem:** Sample output values differ significantly between V1, V2 implementations, and V3.
*   **Status:** Observed. Investigation needed.
*   **Impact:** Complicates correctness verification and comparison.
*   **Research-Informed Potential Causes:**
    *   **Floating-Point Arithmetic:** Parallel execution changes the order of operations (e.g., summation in convolution). Since floating-point math is not strictly associative (`(a+b)+c != a+(b+c)`), minor differences are expected. However, *large* differences suggest bugs.
    *   **Implementation Bugs:** Subtle errors in V2 halo/trimming logic or V3/V4 CUDA kernel indexing could cause significant deviations.
    *   **Initialization:** If random initialization of weights/biases isn't *identical* across all ranks/versions (e.g., different seeds, different sequences), outputs will naturally diverge. Use fixed values or ensure identical seeding for debugging.
    *   **Library Differences (Less Likely Here):** Use of different math libraries or compiler optimization levels could theoretically cause small variations (e.g., `fmaf` usage).
*   **Recommended Diagnostic Steps:**
    1.  **Standardize Inputs:** Use identical, fixed input data and initial weights/biases for all tests.
    2.  **Compare Layer-by-Layer:** Instrument code to save/print the output of *each layer* (Conv1, Pool1, Conv2, etc.) for V1, V2(NP=1), V3, V4(NP=1). Compare outputs at each stage to find where they first diverge significantly.
    3.  **Epsilon Comparison:** Use a tolerance (e.g., `fabs(a - b) < 1e-5`) when comparing floating-point values.
    4.  **Debug Diverging Layer:** Focus debugging efforts on the implementation of the first layer showing significant divergence.

---

## 3. Anticipated Future Problems & Challenges

This section lists potential difficulties or areas requiring significant effort in subsequent project phases.

### V4 Optimization: Asynchronous Overlap Implementation
*   **Challenge:** Effectively overlapping H<->D transfers, GPU computation, and potentially MPI communication requires careful management of CUDA streams and events, and MPI non-blocking calls.
*   **Anticipated Problems:**
    *   **Pinned Memory Requirement:** Asynchronous `cudaMemcpyAsync` *requires* pinned host memory (`cudaMallocHost`), necessitating changes to buffer allocation.
    *   **Synchronization Complexity:** Ensuring correct dependencies (e.g., H->D copy finishes before kernel uses data, kernel finishes before D->H copy starts) using `cudaStreamWaitEvent`, `cudaEventRecord`. Coordinating CUDA events with `MPI_Wait`/`Test` for hybrid overlap.
    *   **Debugging Asynchrony:** Identifying races or incorrect synchronization in asynchronous code is significantly harder than in synchronous code. Tools like Nsight Systems become even more critical.

### V4 Optimization: Host-Side Trimming Logic
*   **Challenge:** The current V4 approach copies the full output tile D->H and trims on the host. This is inefficient.
*   **Anticipated Problems:** Moving trimming to the GPU requires either writing a custom CUDA kernel to extract the valid sub-region or using libraries like Thrust (`thrust::copy_if` with appropriate indexing) before the D->H copy. Alternatively, adjusting kernel launch parameters to only compute the valid output region might be possible but complicates kernel logic.

### V5 CUDA-Aware MPI: Configuration & Performance Validation
*   **Challenge:** Setting up and verifying a truly CUDA-aware MPI environment.
*   **Anticipated Problems:**
    *   *Build/Installation:* Open MPI needs to be compiled with specific flags (`--with-cuda`). Dependencies like UCX might be required and need configuration.
    *   *Environment Variables:* Setting variables like `UCX_TLS=cuda_copy,cuda_ipc` might be needed to enable GPU communication paths.
    *   *Hardware Limitations:* Achieving true GPUDirect RDMA requires specific GPU, NIC, and interconnect hardware/drivers. Performance might fall back to less efficient paths (GPU->Host->Net->Host->GPU) without notice if RDMA is unavailable.
    *   *Verification:* Need specific tests (e.g., OSU micro-benchmarks like `osu_latency -d cuda`, vendor tools) to confirm direct GPU communication is actually occurring and measure its performance benefit over V4's host staging. Small message latency might even increase with CUDA-aware MPI in some cases.

### Advanced CUDA Kernel Optimization
*   **Challenge:** Moving beyond basic CUDA kernels to achieve performance comparable to optimized libraries like cuDNN.
*   **Anticipated Problems:** Requires deep understanding of GPU architecture (SMs, warps, memory hierarchy, caches), advanced techniques like shared memory tiling for convolution (complex indexing, bank conflict avoidance), instruction-level optimization (using `float4` vector types, PTX analysis), occupancy tuning, and extensive profiling with Nsight Compute. Significant effort required per kernel.

### Scalability Limits & Amdahl's Law
*   **Challenge:** Even with efficient parallelization, speedup will eventually be limited by serial portions of the code (Amdahl's Law) or communication overhead that doesn't scale perfectly (e.g., latency in halo exchange).
*   **Anticipated Problems:** Observing diminishing returns in speedup as NP increases for V2.2/V4/V5. Requires careful analysis (strong vs. weak scaling, communication vs. computation breakdown) to understand the limiting factors. The small problem size (Blocks 1&2 only) might lead to communication overhead dominating quickly.

### Target Cluster Environment Variations
*   **Challenge:** Ensuring code and performance are consistent and optimal on the target Fedora cluster, which may differ from the WSL2 development environment.
*   **Anticipated Problems:** Differences in CPU/GPU models, memory capacity/bandwidth, network interconnect (Ethernet vs. InfiniBand), MPI library versions/configurations, CUDA driver versions, OS settings, and scheduler interactions can all affect execution and performance. Re-tuning or debugging might be necessary on the cluster.

### Full AlexNet Extension (Optional)
*   **Challenge:** Implementing FC layers efficiently in parallel and integrating them with the Conv layer parallelization strategy.
*   **Anticipated Problems:** FC layers are typically memory-bandwidth bound. Efficient parallel matrix multiplication (GEMM) is needed (ideally using cuBLAS). Distributing weights/activations for FC layers across MPI ranks requires different communication patterns (e.g., Allreduce for weight gradients if training, potentially different data distributions for inference) than the spatial decomposition used for Conv layers.

---