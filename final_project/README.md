# CS485: GPU Cluster Programming (MPI+CUDA) – Final Project: AlexNet Inference

This directory contains the code, documentation, and resources for the final project of the CS485 GPU Cluster Programming course: an evolving MPI+CUDA implementation of AlexNet inference. The project follows a staged development approach, progressing from serial execution to advanced hybrid parallelism.

**Project Scope Clarification:** The primary focus of the V1-V5 implementation plan is on the **initial two blocks** of the AlexNet architecture (Conv1->ReLU->Pool1 and Conv2->ReLU->Pool2->LRN2). This provides a representative and computationally significant workload for learning and comparing parallelization techniques (MPI, CUDA, Hybrid). Implementing the *full* AlexNet network (including Conv3-5, FC6-8, Softmax) is considered an extension task to be undertaken only *after* the successful completion and analysis of V1-V5 for the initial subset, if time permits.

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Target Environment](#2-target-environment)
3.  [Repository Structure (This Directory)](#3-repository-structure-this-directory)
4.  [Project Version Implementation Plan (Blocks 1 & 2 Focus)](#4-project-version-implementation-plan-blocks-1--2-focus)
    *   [Version 1: Serial CPU](#version-1-serial-cpu---completed)
    *   [Version 2: MPI Only (CPU)](#version-2-mpi-only-cpu---completed)
        *   [Approach 2.1: Broadcast All](#approach-21-broadcast-all---implemented)
        *   [Approach 2.2: Scatter + Halo](#approach-22-scatter--halo---implemented)
    *   [Version 3: CUDA Only (Single GPU)](#version-3-cuda-only-single-gpu---completed)
    *   [Version 4: MPI + CUDA (Hybrid)](#version-4-mpi--cuda-hybrid---pending)
    *   [Version 5: CUDA-Aware MPI (Optional Optimization)](#version-5-cuda-aware-mpi-optional-optimization---pending)
5.  [Key Technologies](#5-key-technologies)
6.  [Current Implementation Status & Performance Highlights](#6-current-implementation-status--performance-highlights)
7.  [Development Workflow for Versions](#7-development-workflow-for-versions)
8.  [Build & Test Instructions per Version](#8-build--test-instructions-per-version)
9.  [Presentation Strategy](#9-presentation-strategy)
10. [Troubleshooting](#10-troubleshooting)
11. [Future Directions & Extensions](#11-future-directions--extensions)
12. [References & Resources](#12-references--resources)

## 1. Project Overview
This project implements inference for the **initial two blocks** of the AlexNet convolutional neural network (Conv1->ReLU->Pool1 and Conv2->ReLU->Pool2->LRN2). The primary goal is to learn and apply different parallel programming paradigms (MPI, CUDA, MPI+CUDA) to this representative workload and to systematically evaluate their performance impact through a structured, incremental 5-version approach. The focus is on the parallelization techniques and performance analysis, rather than building a complete end-to-end classifier.

**Core Task:** Implement and benchmark inference for AlexNet Blocks 1 & 2 across different parallelization stages (V1-V5).

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
├── v1_serial/ # V1: Serial CPU implementation (COMPLETE)
│ ├── include/
│ ├── src/
│ └── Makefile
├── v2_mpi_only/ # V2: MPI-only (CPU cores) implementation (COMPLETE)
│ ├── 2.1_broadcast_all/ # -> Approach 2.1 (Implemented)
│ │ ├── include/
│ │ ├── src/
│ │ └── Makefile
│ └── 2.2_scatter_halo/ # -> Approach 2.2 (Implemented)
│ ├── include/
│ ├── src/
│ └── Makefile
├── v3_cuda_only/ # V3: CUDA-only (single GPU) implementation (COMPLETE)
│ ├── include/
│ ├── src/
│ └── Makefile
├── v4_mpi_cuda/ # V4: Baseline MPI + CUDA implementation (PENDING - Restored Baseline)
│ ├── include/
│ ├── src/
│ └── Makefile
├── v5_cuda_aware_mpi/ # V5: Optional CUDA-aware MPI optimization (PENDING - Restored Baseline)
│ ├── include/
│ ├── src/
│ └── Makefile
├── data/ # SHARED: Input data, model weights, etc. (Accessed via ../data/ or ../../data/)
├── docs/ # SHARED: Project documentation, design notes (Accessed via ../docs/ or ../../docs/)
├── logs/ # Basic run logs (gitignored)
├── logs_extended/ # Detailed performance logs (gitignored)
├── original_backup_*/ # Backup of initial V4/V5 state (gitignored)
├── Makefile.original_v4_v5 # Reference Makefile for V4/V5 base
├── ai_context.txt # Technical context summary for AI assistant
├── discussion.md # Rolling log for professor meetings
├── project.txt # Concatenated source code dump (generated by script)
├── RESEARCH.md # Research findings, critical analysis, references
└── README.md # This file
```

**Note:** Shared resources (`data/`, `docs/`) are accessed from within versioned source code using relative paths. Adjust paths depending on whether accessing from `vX/` or `v2_mpi_only/X.Y/`.

## 4. Project Version Implementation Plan (Blocks 1 & 2 Focus)

The project progresses through five versions, focusing on Blocks 1&2 of AlexNet. V2 explored two distinct MPI strategies. The core goal is to demonstrate understanding of each parallelization paradigm using this subset.

---
### Version 1: Serial CPU - COMPLETED
*   **Directory:** `final_project/v1_serial/`
*   **Goal:** Correct, sequential implementation on a single CPU core. Established functional baseline.
*   **Implementation:** Pure C++, `std::vector`, direct loops for layers.

---
### Version 2: MPI Only (CPU) - COMPLETED
*   **Goal:** Parallelize V1 logic across multiple CPU cores using MPI.
*   **Directory Structure:** Contains subdirectories for implemented approaches.

#### Approach 2.1: Broadcast All - IMPLEMENTED
*   **Directory:** `final_project/v2_mpi_only/2.1_broadcast_all/`
*   **Strategy:** Rank 0 `MPI_Bcast`s full input/parameters. All ranks compute the full V1 layer sequence locally. Each rank extracts its assigned slice from the *final* output only. Rank 0 gathers slices via `MPI_Gatherv`.
*   **Outcome:** Simple implementation, validated basic MPI communication, but demonstrated poor scalability due to broadcast overhead and redundant computation (performance degraded with more processes). Serves as a contrast to more scalable methods.

#### Approach 2.2: Scatter + Halo - IMPLEMENTED
*   **Directory:** `final_project/v2_mpi_only/2.2_scatter_halo/`
*   **Strategy:** Rank 0 `MPI_Scatterv`s input rows. Ranks exchange halo regions using non-blocking `MPI_Isend`/`MPI_Irecv`/`MPI_Wait` before convolution layers. Parameters are broadcast. Each rank computes only on its local data (+halos). Rank 0 gathers final results via `MPI_Gatherv`. Required careful index management and handling of boundary conditions/padding.
*   **Outcome:** More complex implementation (halo management, asymmetric trimming), but demonstrated expected speedup and better scalability compared to 2.1. Represents a more realistic MPI parallelization pattern for convolutions.

---
### Version 3: CUDA Only (Single GPU) - COMPLETED
*   **Directory:** `final_project/v3_cuda_only/`
*   **Goal:** Port V1 compute logic (layers) to run on a single GPU using CUDA. Establish GPU baseline performance.
*   **Implementation:** Basic CUDA kernels implemented for Conv, ReLU, Pool, LRN (e.g., 1D grid-stride loops). Host code manages `cudaMalloc`/`cudaMemcpy`/`cudaFree`, kernel launches. Basic `CUDA_CHECK` error handling used. Uses `nvcc` compiler.
*   **Outcome:** Functional GPU implementation. Initial performance analysis indicates potential bottlenecks likely related to host-device transfer overhead or unoptimized kernels (requires profiling).

---
### Version 4: MPI + CUDA (Hybrid) - PENDING
*   **Directory:** `final_project/v4_mpi_cuda/`
*   **Goal:** Combine MPI parallelism (inter-node/rank, likely based on V2.2 Scatter+Halo logic) with CUDA parallelism (intra-node/rank GPU kernel execution from V3).
*   **Planned Actions:**
    1.  Start from restored baseline code, heavily referencing V2.2 and V3 implementations.
    2.  MPI handles overall structure: data distribution (Scatterv to host buffers), halo communication (Send/Recv on host buffers), result aggregation (Gatherv from host buffers).
    3.  Implement explicit Host<->Device data transfers (`cudaMemcpy` H2D after MPI receive/before kernel, `cudaMemcpy` D2H before MPI send/gather) - **Host Staging**.
    4.  Launch V3 CUDA kernels on device data within each rank.
    5.  Ensure correct GPU affinity (`cudaSetDevice`).
    6.  Implement synchronization between MPI and CUDA operations (`MPI_Wait`, `cudaDeviceSynchronize` or Events).
    7.  Profile to identify bottlenecks (likely host staging, MPI comm).
    8.  Build with `nvcc -ccbin=mpicxx`.

---
### Version 5: CUDA-Aware MPI (Optional Optimization) - PENDING
*   **Directory:** `final_project/v5_cuda_aware_mpi/`
*   **Goal:** Optimize V4 by using CUDA-aware MPI calls (passing GPU device pointers directly to MPI) to potentially reduce/eliminate host staging overhead.
*   **Planned Actions:**
    1.  Modify V4 code's `MPI_Scatterv`, `MPI_Isend`, `MPI_Irecv`, `MPI_Gatherv` calls to use *device* pointers.
    2.  Remove explicit H<->D `cudaMemcpy` calls related solely to MPI staging.
    3.  Verify cluster support (CUDA-aware OpenMPI build, HW support like GPUDirect RDMA). Configure environment if needed (e.g., UCX variables).
    4.  Compare performance against V4 to quantify benefit/overhead.
    5.  Build with `nvcc -ccbin=mpicxx`.

## 5. Key Technologies
- **MPI (Open MPI):** Distributed memory communication.
- **CUDA (NVIDIA):** GPU programming (`nvcc`, runtime API, kernels).
- **C++11:** Host code logic, `std::vector`.
- **Make:** Build system.
- **Bash:** Automation scripts (testing, packaging, scaffolding).

## 6. Current Implementation Status & Performance Highlights

*   **V1 (Serial):** **COMPLETED**. Runs successfully. Baseline time: **~617 ms**.
*   **V2 (MPI - 2.1 Broadcast):** **COMPLETED**. Runs (np=1,2,4). **Degraded performance** (np1:~660ms -> np4:~802ms).
*   **V2 (MPI - 2.2 Scatter+Halo):** **COMPLETED**. Runs (np=1,2,4). **Speedup observed** (np1:~491ms -> np4:~177ms).
*   **V3 (CUDA Only):** **COMPLETED**. Runs (np=1). Initial performance **~750 ms**. Needs profiling (Potential H<->D or kernel bottleneck).
*   **V4 (MPI+CUDA):** **PENDING** (Baseline code restored).
*   **V5 (CUDA-Aware):** **PENDING** (Baseline code restored).

*(Performance times recorded on local WSL2 development machine (CPU: [Your CPU], GPU: [Your GPU]) and require re-evaluation on the target Fedora cluster)*

## 7. Development Workflow for Versions
1.  **Choose Version/Approach:** Select the target (e.g., V4).
2.  **Navigate:** `cd final_project/vX_suffix[/Y.Z_approach]`
3.  **Implement:** Modify code in `src/` and `include/` based on the version's goal, likely starting from the previous validated stage.
4.  **Update Makefile:** Adjust compiler, flags, libraries, and source file list (`SRCS`).
5.  **Build:** Run `make clean && make` within the version directory.
6.  **Test:** Execute the compiled `template` executable (appropriate command for serial/MPI/CUDA).
7.  **Analyze/Profile:** Use timers (`MPI_Wtime`, CUDA Events) and profiling tools (Nsight Systems/Compute) to understand performance bottlenecks.
8.  **Commit:** Save changes frequently using Git.

## 8. Build & Test Instructions per Version

**(Run commands from within the respective subdirectory)**

*   **V1 (Serial):** `cd ../v1_serial && make clean && make && ./template`
*   **V2 (MPI - Broadcast All):** `cd ../v2_mpi_only/2.1_broadcast_all && make clean && make && mpirun -np <N> ./template`
*   **V2 (MPI - Scatter+Halo):** `cd ../v2_mpi_only/2.2_scatter_halo && make clean && make && mpirun -np <N> ./template`
*   **V3 (CUDA Only):** `cd ../v3_cuda_only && make clean && make && ./template`
*   **V4 (MPI+CUDA):** `cd ../v4_mpi_cuda && make clean && make && mpirun -np <N> ./template`
*   **V5 (CUDA-Aware MPI):** `cd ../v5_cuda_aware_mpi && make clean && make && mpirun -np <N> ./template` *(Requires correctly configured CUDA-aware MPI environment)*

*(Use `--oversubscribe` for `mpirun` if testing locally with N > physical cores. On cluster, use `-hostfile` and mapping options.)*

## 9. Presentation Strategy

The final project presentation will focus on the **journey of parallelization and performance analysis using the first two blocks of AlexNet** as the consistent workload. Key elements will include:
*   **Demonstration:** Show V1-V5 (or highest completed version) running successfully.
*   **Methodology:** Explain the parallelization techniques applied in each distinct version (Serial, MPI Broadcast, MPI Scatter+Halo, CUDA, MPI+CUDA, CUDA-Aware MPI). Justify design choices.
*   **Performance Analysis:** Present comprehensive results:
    *   Wall-clock times for each version.
    *   Speedup and Efficiency plots relative to V1.
    *   Scalability analysis (strong scaling) for MPI/Hybrid versions (V2.x, V4, V5).
    *   Timing breakdowns (computation vs communication vs H<->D transfer) using profiler data to identify bottlenecks in each stage.
*   **Conclusion:** Summarize key learnings, challenges overcome, and the effectiveness of each parallelization approach for this specific workload.

## 10. Troubleshooting
*(Section content remains the same - standard troubleshooting tips for Make, Includes, Linking, MPI Runtime, CUDA Runtime, Paths)*
- Makefile Errors: Check for TABs vs spaces...
- Include Errors: Verify include paths...
- Linker Errors: Check linked libraries...
- MPI Runtime Errors: Check `mpirun` syntax...
- CUDA Errors: Use `CUDA_CHECK` macro...
- Path Errors: Double-check relative paths...

## 11. Future Directions & Extensions
- **Complete V4/V5:** Finalize the hybrid implementations and perform thorough benchmarking.
- **Performance Optimization:** Apply advanced techniques (async overlap, kernel tuning, shared memory) to V3/V4/V5 based on profiling results.
- **Full AlexNet Implementation:** *If time permits after V1-V5*, extend the most performant version (likely V4/V5) to include Conv3-5, FC6-8, and Softmax layers. This involves implementing FC layers (matrix multiplication) and potentially adjusting parallel strategies.
- **Distributed Training:** Explore gradient synchronization (`MPI_Allreduce`, potentially via NCCL) as a significantly more advanced topic beyond inference.
- **Explore Alternative MPI Strategies:** Implement and compare V2.2 (Scatter+Halo) within the V4/V5 hybrid context.

## 12. References & Resources
- MPI Forum: [mpi-forum.org](https://mpi-forum.org/)
- NVIDIA CUDA Documentation: [docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
- Programming Massively Parallel Processors (4th Ed.) Textbook & Companion Site
- Open MPI Documentation: [open-mpi.org](https://www.open-mpi.org/)
- LLNL HPC Tutorials: [hpc-tutorials.llnl.gov](https://hpc-tutorials.llnl.gov/)
- `final_project/RESEARCH.md` for detailed analysis and specific paper references.