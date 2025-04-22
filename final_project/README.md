# CS485: GPU Cluster Programming (MPI+CUDA) – Project Workspace
# CS485: GPU Cluster Programming (MPI+CUDA) – Final Project: AlexNet Inference

This directory contains the code, documentation, and resources for the final project of the CS485 GPU Cluster Programming course: an evolving MPI+CUDA implementation of AlexNet inference. The project follows a staged development approach, progressing from serial execution to advanced hybrid parallelism.

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Target Environment](#2-target-environment)
3.  [Repository Structure (This Directory)](#3-repository-structure-this-directory)
4.  [Project Version Implementation Plan](#4-project-version-implementation-plan)
    *   [Version 1: Serial CPU](#version-1-serial-cpu)
    *   [Version 2: MPI Only (CPU)](#version-2-mpi-only-cpu)
    *   [Version 3: CUDA Only (Single GPU)](#version-3-cuda-only-single-gpu)
    *   [Version 4: MPI + CUDA (Hybrid)](#version-4-mpi--cuda-hybrid)
    *   [Version 5: CUDA-Aware MPI (Optional Optimization)](#version-5-cuda-aware-mpi-optional-optimization)
5.  [Key Technologies](#5-key-technologies)
6.  [Current Implementation Status](#6-current-implementation-status)
7.  [Development Workflow for Versions](#7-development-workflow-for-versions)
8.  [Build & Test Instructions per Version](#8-build--test-instructions-per-version)
9.  [Troubleshooting](#9-troubleshooting)
10. [Future Directions](#10-future-directions)
11. [References & Resources](#11-references--resources)

## 1. Project Overview
This project implements inference for the initial blocks of the AlexNet convolutional neural network. The goal is to explore different parallel programming paradigms (MPI, CUDA, MPI+CUDA) and evaluate their performance on a GPU cluster. The implementation follows a structured, incremental approach across five distinct versions.

**Core Task:** Implement AlexNet inference, starting with:
  *   Block 1: Conv1 → ReLU1 → Pool1
  *   Block 2: Conv2 → ReLU2 → Pool2 → LRN2

**Parallelization Strategy (V4/V5):** Primarily data parallelism, where input data batches are distributed across MPI ranks, each rank utilizing its assigned GPU(s) for computation. Model weights are typically broadcast from rank 0.

## 2. Target Environment
Code must ultimately compile and run correctly under the course's specified environment:
- **OS:** Fedora 37
- **Compilers:** GCC 12 (for host code), `mpicc`/`mpicxx` (MPI wrappers), `nvcc` (CUDA compiler)
- **GPU Toolkit:** CUDA 12.x
- **MPI:** Open MPI (ideally compiled with CUDA-awareness for V5)

Local development is done in WSL2 (Ubuntu) with compatible toolchains, but final testing should target the Fedora environment.

## 3. Repository Structure (This Directory)

```
final_project/
├── v1_serial/ # Version 1: Serial CPU implementation
│ ├── include/ # Headers specific to V1 (modified from base)
│ ├── src/ # Source code specific to V1 (modified from base)
│ └── Makefile # Makefile for V1 (using g++)
├── v2_mpi_only/ # Version 2: MPI-only (CPU cores) implementation
│ ├── include/ # Headers specific to V2
│ ├── src/ # Source code specific to V2
│ └── Makefile # Makefile for V2 (using mpicxx)
├── v3_cuda_only/ # Version 3: CUDA-only (single GPU) implementation
│ ├── include/ # Headers specific to V3
│ ├── src/ # Source code specific to V3
│ └── Makefile # Makefile for V3 (using nvcc)
├── v4_mpi_cuda/ # Version 4: Baseline MPI + CUDA implementation
│ ├── include/ # Headers for V4 (restored from backup)
│ ├── src/ # Source code for V4 (restored from backup)
│ └── Makefile # Makefile for V4 (using nvcc -ccbin=mpicxx)
├── v5_cuda_aware_mpi/ # Version 5: Optional CUDA-aware MPI optimization
│ ├── include/ # Headers for V5 (likely same as V4)
│ ├── src/ # Source code for V5 (modified V4 for direct GPU pointers in MPI)
│ └── Makefile # Makefile for V5 (likely same as V4)
├── data/ # SHARED: Input data, model weights, etc. (Accessed via ../data/)
├── docs/ # SHARED: Project documentation, design notes (Accessed via ../docs/)
├── logs/ # Basic run logs (gitignored)
├── logs_extended/ # Detailed performance logs (gitignored)
├── original_backup_*/ # Backup of initial V4/V5 state (gitignored)
├── Makefile.original_v4_v5 # Reference Makefile for V4/V5 base
└── README.md # This file
```

**Note:** Shared resources (`data/`, `docs/`) are accessed from within versioned source code using relative paths (e.g., `../data/imagenet.batch`).

## 4. Project Version Implementation Plan

The project progresses through five versions. Initial setup (via `enhance_scaffold_all_versions.sh`) populates V1-V3 with code from V4/V5 and attempts basic automated commenting/renaming. **Significant manual effort is required for each version.**

**General Manual Steps:**
1.  Navigate to the target version directory (`cd final_project/vX_suffix`).
2.  Carefully review code in `src/` and `include/`, especially lines commented with `// MPI?` or `// CUDA?`. Delete/modify as needed.
3.  Implement the core logic required for the version (e.g., serial CPU replacements for CUDA kernels).
4.  Clean up includes and header file prototypes.
5.  Verify/update the `Makefile`, **critically updating the `SRCS` list** (for V1-V3) to match final source file names.
6.  Ensure all paths accessing shared resources use the correct relative path (e.g., `../data/`).

---
### Version 1: Serial CPU
*   **Directory:** `final_project/v1_serial/`
*   **Goal:** Correct, purely sequential AlexNet inference on a single CPU core. Establish functional baseline.
*   **Manual Actions:**
    *   **Remove MPI:** Delete `MPI_` calls, includes.
    *   **Remove CUDA:** Delete kernel calls, memory management (`cudaMalloc`, `cudaMemcpy`, etc.), CUDA types/includes (`__global__`, `cudaStream_t`, `cuda_runtime.h`).
    *   **Implement Serial Logic:** Replace *all* CUDA kernel functionality (Conv, ReLU, Pool, LRN) with standard C++ loops and calculations in the relevant `.cpp` files (rename `.cu` files to `.cpp` after removing CUDA code).
    *   **Makefile:** Use `g++`. Update `SRCS` list with final `.cpp` filenames. Remove MPI/CUDA flags/libs.
    *   **Paths:** Verify `../data/` access.

---
### Version 2: MPI Only (CPU)
*   **Directory:** `final_project/v2_mpi_only/`
*   **Goal:** Parallelize the serial logic across multiple CPU cores on potentially multiple nodes using MPI for data distribution.
*   **Manual Actions:**
    *   **Keep MPI:** Ensure `MPI_Init`, `MPI_Comm_*`, `MPI_Bcast`, `MPI_Allreduce` (or Scatter/Gather), `MPI_Finalize` are correctly used for data parallelism.
    *   **Remove CUDA:** Delete kernel calls, CUDA memory management, types/includes.
    *   **Implement Serial Logic per Rank:** Replace CUDA kernel functionality with standard C++ code that runs *within each MPI rank* on its assigned data slice. (Rename `.cu` files to `.cpp`).
    *   **Makefile:** Use `mpicxx`. Update `SRCS` list with final `.cpp` filenames. Remove CUDA flags/libs.
    *   **Paths:** Verify `../data/` access.

---
### Version 3: CUDA Only (Single GPU)
*   **Directory:** `final_project/v3_cuda_only/`
*   **Goal:** Port the computationally intensive parts (layers) to run on a single GPU using CUDA.
*   **Manual Actions:**
    *   **Remove MPI:** Delete `MPI_` calls, includes. Program runs as a single process.
    *   **Keep/Refine CUDA:** Ensure CUDA kernels, `cudaMalloc`, `cudaMemcpy`, `cudaEvent` or `cudaDeviceSynchronize` are correct for processing the *entire* dataset on one GPU. Keep relevant `.cu` files.
    *   **Makefile:** Use `nvcc`. Update `SRCS_CU`/`SRCS_CPP` lists. Ensure correct `-gencode` flags for target GPU and link `-lcudart`. Remove MPI wrapper (`-ccbin`).
    *   **Paths:** Verify `../data/` access.

---
### Version 4: MPI + CUDA (Hybrid)
*   **Directory:** `final_project/v4_mpi_cuda/`
*   **Goal:** Combine MPI parallelism (inter-node) with CUDA parallelism (intra-node GPUs). This is the baseline hybrid version restored from backup.
*   **Manual Actions:**
    *   **Verify Code:** Ensure MPI calls correctly manage data distribution and synchronization around CUDA operations. Check host-device data transfers (`cudaMemcpy`) needed for communication.
    *   **Verify Makefile:** Should use `nvcc -ccbin=mpicxx` (or equivalent MPI C++ wrapper). Check CUDA flags (`-gencode`) and linked libraries (`cudart`). Verify `SRCS_CU`/`SRCS_CPP` lists are correct.
    *   **Paths:** Verify `../data/` access.

---
### Version 5: CUDA-Aware MPI (Optional Optimization)
*   **Directory:** `final_project/v5_cuda_aware_mpi/`
*   **Goal:** Optimize V4 by using CUDA-aware MPI features to allow direct communication using GPU device pointers, reducing host-device copies.
*   **Manual Actions:**
    *   **Modify MPI Calls:** Change relevant MPI calls (e.g., `MPI_Bcast`, `MPI_Allreduce`, `MPI_Send`, `MPI_Recv`) to pass **GPU device memory pointers** instead of host pointers. This requires an MPI library built with CUDA support and correctly configured environment.
    *   **Remove Redundant Copies:** Eliminate `cudaMemcpy` calls that were previously needed solely to stage data for MPI communication.
    *   **Verify Makefile:** Likely identical to V4 (`nvcc -ccbin=mpicxx`). Correctness depends on the runtime MPI library's CUDA support, not typically compile-time flags specific to this feature.
    *   **Paths:** Verify `../data/` access.

## 5. Key Technologies
- **MPI (Open MPI):** Distributed memory communication.
- **CUDA (NVIDIA):** GPU programming (`nvcc`, runtime API, kernels).
- **C++:** Host code logic.
- **Make:** Build system.
- **Bash:** Automation scripts.

## 6. Current Implementation Status
*(Update this section as you progress)*
- **V1 (Serial):** [Not Started / In Progress / Completed]
- **V2 (MPI Only):** [Not Started / In Progress / Completed]
- **V3 (CUDA Only):** [Not Started / In Progress / Completed]
- **V4 (MPI+CUDA):** Base implementation restored from backup. Assumed working.
- **V5 (CUDA-Aware):** [Not Started / In Progress / Completed]

**Implemented Layers (in V4 base):**
  1. Conv1 → ReLU1 → Pool1
  2. Conv2 → ReLU2 → Pool2 → LRN2

## 7. Development Workflow for Versions
1.  **Choose Version:** Select the version (V1-V5) to work on.
2.  **Navigate:** `cd final_project/vX_suffix`
3.  **Implement:** Modify code in `src/` and `include/` according to the plan in [Section 4](#4-project-version-implementation-plan).
4.  **Update Makefile:** Adjust compiler, flags, libraries, and **source file list (`SRCS`)** as needed.
5.  **Build:** Run `make clean && make` within the version directory.
6.  **Test:** Execute the compiled `template` executable (e.g., `./template` for V1/V3, `mpirun -np X ./template` for V2/V4/V5).
7.  **Commit:** Save changes frequently using Git.

## 8. Build & Test Instructions per Version

**(Run commands from within the respective `final_project/vX_suffix` directory)**

*   **V1 (Serial):**
    ```bash
    # (Inside final_project/v1_serial/)
    make clean && make
    ./template [optional_args]
    ```
*   **V2 (MPI Only):**
    ```bash
    # (Inside final_project/v2_mpi_only/)
    make clean && make
    mpirun -np <num_processes> ./template [optional_args]
    ```
*   **V3 (CUDA Only):**
    ```bash
    # (Inside final_project/v3_cuda_only/)
    make clean && make
    ./template [optional_args]
    ```
*   **V4 (MPI+CUDA):**
    ```bash
    # (Inside final_project/v4_mpi_cuda/)
    make clean && make
    mpirun -np <num_processes> ./template [optional_args]
    # Use --oversubscribe locally if needed
    # On cluster, use -hostfile and potentially --map-by node/socket/gpu
    ```
*   **V5 (CUDA-Aware MPI):**
    ```bash
    # (Inside final_project/v5_cuda_aware_mpi/)
    # Requires MPI library compiled with CUDA support!
    make clean && make
    mpirun -np <num_processes> ./template [optional_args]
    # Ensure environment variables (e.g., UCX settings) are correct if using UCX backend
    ```

## 9. Troubleshooting
- **Makefile Errors:** Check for TABs vs spaces, correct compiler (`g++`, `mpicxx`, `nvcc`), valid flags, correctly listed source files (`SRCS=`).
- **Include Errors:** Verify include paths (`-I./include`).
- **Linker Errors:** Check linked libraries (`-lm`, `-lcudart`). For V4/V5, ensure `nvcc` uses the MPI wrapper via `-ccbin=mpicxx`.
- **MPI Runtime Errors:** Check `mpirun` syntax, host connectivity, firewall settings. Use `--oversubscribe` for local testing beyond core count.
- **CUDA Errors:** Use `cudaGetLastError()`/`cudaDeviceSynchronize()` after kernel calls and `cudaMemcpy`. Check memory allocations and grid/block dimensions. Ensure target GPU supports chosen compute capability (`-gencode arch=compute_XX,code=sm_XX`).
- **Path Errors:** Double-check relative paths (`../data/`) in source code.

## 10. Future Directions
- Implement remaining AlexNet layers (Conv3-5, Fully Connected, Softmax).
- Optimize CUDA kernels (tiling, shared memory, instruction-level parallelism).
- Explore different parallelization strategies (model parallelism).
- Integrate performance profiling tools (nvprof, Nsight Systems/Compute).
- Implement distributed training (requires gradient synchronization).

## 11. References & Resources
- MPI Forum: [mpi-forum.org](https://mpi-forum.org/)
- NVIDIA CUDA Documentation: [docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- Programming Massively Parallel Processors (4th Ed.) Textbook & Companion Site
- Open MPI Documentation: [open-mpi.org](https://www.open-mpi.org/)
- LLNL HPC Tutorials: [hpc-tutorials.llnl.gov](https://hpc-tutorials.llnl.gov/)