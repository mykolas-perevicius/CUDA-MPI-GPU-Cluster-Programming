# CS485 Final Project: Research, Analysis, and Findings

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation - Blocks 1&2)
**Purpose:** This document captures research findings, critical analysis of the project plan, High-Performance Computing (HPC) best practices, and relevant context from academic literature and technical documentation pertaining to the CS485 final project. It serves as a rolling reference to inform design decisions and anticipate challenges throughout the V1-V5 implementation stages.

---

## Table of Contents

1.  [Executive Summary of Analysis](#1-executive-summary-of-analysis)
2.  [Pedagogical Evaluation of Staged (V1-V5) Approach](#2-pedagogical-evaluation-of-staged-v1-v5-approach)
    *   [Analysis of the Learning Curve](#21-analysis-of-the-learning-curve)
    *   [Common Student Challenges](#22-common-student-challenges)
    *   [Alignment with Course Objectives](#23-alignment-with-course-objectives)
3.  [Development Environment and Build System](#3-development-environment-and-build-system)
    *   [WSL2 vs. Native Linux (Fedora): Compatibility and Performance](#31-wsl2-vs-native-linux-fedora-compatibility-and-performance)
    *   [Build System: Makefiles and Bash Scripting](#32-build-system-makefiles-and-bash-scripting)
4.  [Analysis of MPI Parallelization Strategies (V2 Focus)](#4-analysis-of-mpi-parallelization-strategies-v2-focus)
    *   [Critique of "Broadcast All, Compute Slices" Approach (V2.1)](#41-critique-of-broadcast-all-compute-slices-approach-v21)
    *   [Analysis of Scatter+Halo Approach (V2.2)](#42-analysis-of-scatterhalo-approach-v22) <!-- Renamed from comparison -->
    *   [Sub-Version Implementation Philosophy](#43-sub-version-implementation-philosophy)
    *   [Correctness Pitfalls in MPI](#44-correctness-pitfalls-in-mpi)
5.  [Layer Parallelization: AlexNet Subset & Challenges (V3 Focus)](#5-layer-parallelization-alexnet-subset--challenges-v3-focus)
    *   [Representativeness of Early AlexNet Blocks](#51-representativeness-of-early-alexnet-blocks)
    *   [Inherent Difficulties in Layer Parallelization (MPI & CUDA)](#52-inherent-difficulties-in-layer-parallelization-mpi--cuda)
    *   [Critique of `std::vector<float>` for HPC/CUDA](#53-critique-of-stdvectorfloat-for-hpccuda)
6.  [CUDA Implementation and MPI+CUDA Integration (V3-V5)](#6-cuda-implementation-and-mpicuda-integration-v3-v5)
    *   [V3 (CUDA Kernels): Best Practices and Pitfalls](#61-v3-cuda-kernels-best-practices-and-pitfalls)
    *   [V4 (MPI+CUDA): Integration Challenges](#62-v4-mpicuda-integration-challenges)
    *   [V5 (CUDA-aware MPI): Requirements, Benefits, Pitfalls](#63-v5-cuda-aware-mpi-requirements-benefits-pitfalls)
7.  [HPC Software Engineering and Development Practices](#7-hpc-software-engineering-and-development-practices)
    *   [Code Organization and Modularity](#71-code-organization-and-modularity)
    *   [Robust Error Handling (MPI & CUDA)](#72-robust-error-handling-mpi--cuda)
    *   [Debugging Techniques per Stage](#73-debugging-techniques-per-stage)
    *   [Data Precision (Single vs. Double)](#74-data-precision-single-vs-double)
8.  [Performance Analysis, Bottlenecks, and Profiling](#8-performance-analysis-bottlenecks-and-profiling)
    *   [Likely Performance Bottlenecks per Stage](#81-likely-performance-bottlenecks-per-stage)
    *   [Standard Performance Metrics](#82-standard-performance-metrics)
    *   [Suitable Profiling Tools/Techniques](#83-suitable-profiling-toolstechniques)
9.  [Advanced Topics, Libraries Context, and Potential Gaps](#9-advanced-topics-libraries-context-and-potential-gaps)
    *   [Gaps in Current Plan (Async Overlap, Load Balancing, etc.)](#91-gaps-in-current-plan-async-overlap-load-balancing-etc)
    *   [Awareness of Relevant Libraries (cuDNN, NCCL, Thrust, etc.)](#92-awareness-of-relevant-libraries-cudnn-nccl-thrust-etc)
    *   [Validation and Numerical Correctness Concerns](#93-validation-and-numerical-correctness-concerns)
10. [Research-Based Recommendations for the Project](#10-research-based-recommendations-for-the-project)
11. [References](#11-references)
12. [Rolling Research Notes & Findings](#12-rolling-research-notes--findings)

---

## 1. Executive Summary of Analysis

The CS485 final project plan, implementing AlexNet inference via a five-stage parallelization (V1-V5), is pedagogically valuable for its incremental introduction to Serial, MPI, CUDA, and Hybrid MPI+CUDA programming. However, it presents significant technical and conceptual challenges. Key concerns identified through research and code review include:
*   **Learning Curve:** Steep difficulty increases between stages (V1->V2, V3->V4). V4 integration is particularly complex.
*   **V2 MPI Strategy:** The "Broadcast All" (V2.1) approach demonstrated poor scaling as expected. The implemented Scatter+Halo (V2.2) is more scalable but involved complex halo/trimming logic.
*   **Environment:** Developing in WSL2 introduces compatibility/performance risks compared to the target Fedora cluster. Observed V4 NP=4 crash could be environment-related.
*   **Data Structures:** `std::vector<float>` remains suboptimal for CUDA H<->D transfers; pinned memory is crucial for optimized V4/V5.
*   **V3 Performance:** The CUDA-only version is significantly slower than serial, pointing to H<->D overhead or kernel inefficiencies needing profiling.
*   **V4 Implementation:** The current host-staging approach (full padded tile H<->D, GPU computes full sequence, D->H, host trim) introduces multiple potential bottlenecks and complexity points. Debugging is currently required (output format, NP=4 crash).
*   **V5 Feasibility:** CUDA-aware MPI benefits depend heavily on specific cluster HW/SW support (GPU Direct RDMA) and may not eliminate all H<->D if host-side logic (like trimming) remains.
*   **Bottlenecks:** Performance limitations shift dramatically (CPU -> MPI Comm -> GPU Kernel/Transfers -> V4 Host Staging/Sync/Comm -> Network).
*   **Software Engineering:** Robust error handling (using `CUDA_CHECK` and MPI checks), modularity, and effective debugging/profiling are critical but challenging, especially in V4.
*   **Context:** Plan lacks emphasis on asynchronous overlap and standard libraries (cuDNN, NCCL). Numerical discrepancies across versions need attention.

Recommendations focus on standardizing the environment, prioritizing V4 debugging/profiling, mandating appropriate data structures by V3, verifying V5 viability, enforcing SE practices, analyzing numerical differences, and contextualizing manual efforts with industry libraries/techniques.

---

## 2. Pedagogical Evaluation of Staged (V1-V5) Approach

### 2.1 Analysis of the Learning Curve
*(Content largely unchanged, still relevant)*
The 5-stage approach offers a structured learning path:
*   **V1 (Serial):** Establishes functional correctness and a performance baseline. Reinforces algorithm understanding.
*   **V2 (MPI):** Introduces distributed memory concepts, explicit communication (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Isend/Irecv`, `MPI_Gather`), halo exchange, and process management. A major conceptual shift.
*   **V3 (CUDA):** Introduces GPU architecture, SIMT execution, kernel programming, and host-device memory management. Another distinct, significant conceptual leap.
*   **V4 (MPI+CUDA):** The core integration challenge, demanding orchestration across nodes and GPUs, managing data movement between host (MPI) and device (CUDA) memory spaces (via host staging in current implementation), and synchronization. Often the steepest curve.
*   **V5 (CUDA-aware MPI):** Optimization layer potentially simplifying V4 communication code and improving performance via direct GPU communication. Requires specific HW/SW support; pedagogical value depends on demonstrable benefits vs V4.

**Overall:** The staging isolates concepts effectively, but instructors/students must anticipate difficulty spikes, especially V1->V2 and V3->V4. V4 debugging is currently a significant hurdle.

### 2.2 Common Student Challenges
*(Content largely unchanged, still relevant)*
*   **Conceptual:** Distinguishing distributed vs. shared memory models. Understanding MPI communicators, CUDA execution/memory hierarchies, synchronization primitives (`MPI_Barrier`, `MPI_Wait`, `cudaDeviceSynchronize`, Events, `__syncthreads`). Correctly mapping algorithm stages to parallel execution models.
*   **Implementation:** Debugging parallel/distributed non-deterministic code. Managing complex C++/MPI/CUDA builds (`nvcc -ccbin`). Correct API usage and error checking (`CUDA_CHECK`). Performance tuning (identifying/addressing bottlenecks like H<->D transfers). Correct halo exchange and data trimming logic.
*   **Integration (V4/V5):** Sequencing MPI calls and CUDA operations correctly. Ensuring host/device data consistency (especially with host staging in V4). Avoiding deadlocks. Managing GPU affinity (`cudaSetDevice`). Configuring/verifying CUDA-aware MPI (V5). Resolving runtime errors (like the V4 NP=4 crash).

### 2.3 Alignment with Course Objectives
*(Content largely unchanged, still relevant)*
This project aligns well with typical HPC course objectives:
*   Hands-on experience with parallelization concepts (data/task parallelism, speedup, efficiency).
*   MPI standard for distributed memory programming.
*   CUDA for GPU accelerator programming.
*   Hybrid MPI+CUDA models (specifically host-staging and potentially direct communication).
*   Performance analysis and profiling.
*   Application to a relevant domain (CNN inference).

---

## 3. Development Environment and Build System

### 3.1 WSL2 vs. Native Linux (Fedora): Compatibility and Performance
*(Content largely unchanged, warning remains highly relevant given V4 crash)*
Using WSL2 (Ubuntu) for development while targeting Fedora 37 presents risks:
*   **WSL2 Overview:** Runs a real Linux kernel via lightweight VM. Good compatibility but not identical to native Linux.
*   **Performance Discrepancies:** Filesystem I/O (Windows<->WSL2) can be slow. Virtualized networking may differ from native MPI performance. GPU compute usually close, but potential overheads exist.
*   **Compatibility Risks:** Differences in kernel versions, system libraries (glibc), CUDA drivers (WSL vs. native), and MPI implementation behavior (e.g., OpenMPI quirks [5], resource limits) can cause "works on my machine" issues. Consumer GPUs in WSL2 may lack features like GPUDirect RDMA [5]. The V4 NP=4 crash *could* be an environment-specific resource issue.
*   **Mitigation:** Minimize environment differences. Use containers (Docker/Singularity) in both dev/target environments. Perform frequent testing on the actual Fedora cluster. Match toolchain versions (GCC 12, CUDA 12.x, Open MPI) precisely.

**Conclusion:** Sole reliance on WSL2 is risky, especially for debugging runtime errors like the V4 crash. Prioritize testing on or containerizing the target environment.

### 3.2 Build System: Makefiles and Bash Scripting
*(Content updated for V4)*
*   **Suitability:** Makefiles + Bash are standard HPC tools. V1-V3 use basic Makefiles. V4 introduces a more complex Makefile integrating `nvcc -ccbin=mpicxx`, MPI flag detection (`--showme`), `bear` for `compile_commands.json`, and `clang-tidy` for linting. This is a good step towards better SE practices but adds complexity.
*   **Complexity Management:** Requires well-organized Makefiles (variables, pattern rules, dependencies). Robust Bash scripts (`run_final_project.sh`) with error checking are used for automation.
*   **Alternative (CMake):** Offers better cross-platform support, dependency finding (FindMPI, FindCUDA), build configurations, and test integration. Scales better for complex projects and is a valuable skill. Represents stronger software engineering practice, though with a slightly steeper initial learning curve than basic Makefiles.

---

## 4. Analysis of MPI Parallelization Strategies (V2 Focus)

### 4.1 Critique of "Broadcast All, Compute Slices" Approach (V2.1)
*(Content largely unchanged, validated by results)*
This strategy (`MPI_Bcast` full input/parameters, each rank computes full V1 sequence, `MPI_Gather`/`MPI_Gatherv` final slices) was implemented and tested.
*   **Scalability Issues:**
    *   *Communication Bottleneck:* `MPI_Bcast` cost scales poorly with process count (P) and data size. Gather also adds overhead. Performance degraded as P increased, as expected.
    *   *Memory Inefficiency:* Each rank stores full input/parameters, negating distributed memory benefits. Limited by single-node memory.
*   **Implementation Simplicity:** Main advantage. Ranks compute independently after broadcast.
*   **Pedagogical Value:** Introduces basic collectives but fails to teach scalable communication patterns. Serves as a useful negative example compared to V2.2.

### 4.2 Analysis of Scatter+Halo Approach (V2.2)
*(Content updated to reflect implementation)*
This strategy (Scatter input rows, exchange halos via `MPI_Isend/Irecv/Wait`, broadcast parameters, compute locally, trim results, Gather output) was implemented and demonstrated good scalability.
*   **Pros:** Reduced activation memory per rank. Good load balancing (with roughly equal row distribution). Neighbor communication more scalable than broadcast. Natural fit for convolution.
*   **Cons:** Higher implementation complexity (halo logic, non-blocking MPI, boundary handling, complex asymmetric trimming after pooling layers). Correctness is harder to achieve.
*   **Key Implementation Points:** Used non-blocking MPI for halo overlap potential. Required careful calculation of which rows constituted halos and which rows needed trimming after pooling layers based on the halo influence propagating through Conv/Pool stages.

### 4.3 Sub-Version Implementation Philosophy
*(Content unchanged)*
Recognize that multiple valid strategies exist for each stage.
1.  **Identify & Document:** Note different approaches in `README.md` or `RESEARCH.md`.
2.  **Implement Primary:** Implement one chosen strategy first within the main version folder (e.g., V2.2 in `v2_mpi_only/2.2_scatter_halo/`).
3.  **Explore Alternatives (Optional):** Implement alternatives in sub-folders (e.g., V2.1) or via Git branches for comparison.

### 4.4 Correctness Pitfalls in MPI
*(Content unchanged, highly relevant to V4 debugging)*
*   **Collectives:** Ensuring all ranks participate, correct counts/displacements/datatypes (`MPI_Scatterv`, `MPI_Gatherv`), correct reduction operations (if used). Off-by-one errors in slicing/distribution.
*   **Halo Exchange (V2.2/V4):** Incorrect neighbor ranks, mismatched tags/counts, buffer errors (size, pointers), deadlocks (blocking send/recv order, insufficient buffer space), race conditions (non-blocking), boundary conditions (rank 0, rank size-1).
*   **Data Types:** `MPI_Datatype` must match C++ memory layout precisely (`MPI_FLOAT` vs `float`).
*   **Buffer Management:** Ensuring send buffers aren't modified before non-blocking sends complete (`MPI_Wait`). Ensuring receive buffers are large enough.

---

## 5. Layer Parallelization: AlexNet Subset & Challenges (V3 Focus)

### 5.1 Representativeness of Early AlexNet Blocks
*(Content unchanged)*
*   **Layers Covered:** Conv1, ReLU1, Pool1, Conv2, ReLU2, Pool2, LRN2.
*   **Characteristics:** Covers compute-intensive convolution, simple element-wise ReLU, local reduction pooling, and more complex cross-channel LRN. Representative of fundamental CNN operations.
*   **Limitations:** Omits deeper Conv layers (smaller spatial size, more channels) and Fully Connected layers (matrix multiplies, different bottlenecks). May give skewed perspective on optimal strategies for the whole network. Early layers have large spatial dimensions suitable for spatial decomposition; later layers might favor filter decomposition.

### 5.2 Inherent Difficulties in Layer Parallelization (MPI & CUDA)
*(Content unchanged)*
*   **MPI:**
    *   *Convolution:* Data distribution needs halo exchange (if spatial decomp.) or input replication (if filter decomp.). Load balancing.
    *   *Pooling:* Simpler, but needs halos if spatial decomp. and window crosses boundary.
    *   *LRN:* Cross-channel/spatial dependencies make efficient distribution hard.
*   **CUDA:**
    *   *Convolution:* Efficient mapping of loops to threads/blocks/grids. Optimizing memory access (coalescing, shared memory tiling for reuse). Algorithms (im2col, Winograd, direct).
    *   *Pooling:* Simpler kernel, boundary checks, avoid shared memory bank conflicts if used.
    *   *LRN:* Efficient access to neighbors (spatial/channel). Shared memory useful but needs careful indexing.

### 5.3 Critique of `std::vector<float>` for HPC/CUDA
*(Content updated - V4 still uses vector on host)*
Using `std::vector<float>` on the host (as done in V1-V4) for data that needs to be transferred to/from the GPU presents performance issues:
*   **Memory Allocation:** Allocates standard pageable host memory.
*   **Host-Device Transfer:** `cudaMemcpy` is significantly slower with pageable memory compared to pinned (page-locked) host memory (allocated via `cudaMallocHost`). Pinned memory enables asynchronous transfers (`cudaMemcpyAsync`) essential for overlapping communication/computation (a potential optimization for V4/V5). The current V4 uses synchronous `cudaMemcpy` with pageable `std::vector` buffers, maximizing transfer latency. [9]
*   **Alignment:** Default alignment may not be optimal for SIMD/GPU coalescing.
*   **Recommendation:** Transition to pinned memory strategies (`cudaMallocHost`) for performance-critical H<->D transfer buffers, especially if attempting asynchronous overlap in V4/V5 optimizations. Raw `float*` with `cudaMallocHost`/`cudaFreeHost` or Thrust vectors are alternatives.

---

## 6. CUDA Implementation and MPI+CUDA Integration (V3-V5)

### 6.1 V3 (CUDA Kernels): Best Practices and Pitfalls
*(Content unchanged - analysis still applies to V3/V4 kernels)*
*   **Best Practices:** Map computation to CUDA hierarchy. Minimize H<->D transfers. Use pinned host memory (not done yet). Optimize global memory access (coalescing). Use shared memory for data reuse (beware bank conflicts). Maximize arithmetic intensity. Avoid warp divergence. Use CUDA streams for overlap (not done yet). Profile with Nsight Compute/Systems. Rigorous error checking (`CUDA_CHECK`).
*   **Pitfalls:** Uncoalesced access. Global memory bottleneck. Shared memory bank conflicts. Low occupancy. Thread divergence. Forgetting error checks. Underestimating transfer overhead (likely cause of V3 slowness). Kernel indexing errors.

### 6.2 V4 (MPI+CUDA): Integration Challenges
*(Content updated based on V4 code)*
*   **Host/Device Data Management (Current V4 Strategy):**
    1.  MPI Scatter to host `std::vector`.
    2.  MPI Halo exchange uses host buffers.
    3.  *Full padded* local slice copied H->D (`cudaMemcpy` from `std::vector` to `cudaMalloc`'d buffer). **Bottleneck 1: Large, synchronous copy.**
    4.  Entire layer sequence (Conv1..LRN2) computed on GPU tile via `alexnetTileForwardCUDA`. This helper likely allocates/copies weights/biases internally on each call. **Potential Bottleneck 2: Redundant internal copies/allocs?**
    5.  Final result tile copied D->H (`cudaMemcpy` to `std::vector`). **Bottleneck 3: Large, synchronous copy.**
    6.  Result trimmed on *host*. **Bottleneck 4: CPU work after GPU, potential serialization.**
    7.  MPI Gather from host `std::vector`.
    This "host staging" pattern is simple conceptually but adds significant latency and PCIe load compared to potentially overlapping or direct communication approaches.
*   **Synchronization:** Current code appears largely synchronous (blocking MPI halo waits, synchronous `cudaMemcpy`, likely implicit sync after kernels before D->H copy). Need to ensure `MPI_Waitall` for halos completes before H->D copy, and GPU work completes (`cudaDeviceSynchronize` likely needed before D->H copy, although `cudaMemcpy` D->H implies sync) before host trimming/gather. Incorrect sync leads to race conditions or deadlocks.
*   **GPU Affinity:** Implemented via `cudaSetDevice(rank % nDev)`. Seems correct but relies on linear rank-to-GPU mapping.
*   **Resource Management:** The NP=4 crash suggests potential issues with memory allocation (host or device limits exceeded with more processes) or MPI resource exhaustion (e.g., message queues, unexpected blocking). Needs debugging.
*   **Complexity:** Managing indices, buffer sizes, padding/trimming logic correctly across MPI ranks and host/device boundaries is error-prone.

### 6.3 V5 (CUDA-aware MPI): Requirements, Benefits, Pitfalls
*(Content updated based on V4)*
*   **Requirements:** MPI library built with CUDA support (OpenMPI `--with-cuda`, MVAPICH2-GDR, etc.). Compatible CUDA Toolkit/drivers. HW support for GPU Direct RDMA (specific GPUs/NICs/fabric) for optimal performance. Correct system/MPI environment configuration (e.g., UCX settings [4, 5]).
*   **Benefits:** Pass GPU device pointers directly to MPI calls (`MPI_Scatterv`, `MPI_Isend`, `Irecv`, `Gatherv`). Library handles transfer, potentially via RDMA bypassing host memory. Reduces latency, PCIe traffic, CPU overhead *for the communication parts*. Simplifies *application* code by removing manual H<->D `cudaMemcpy` calls *for MPI staging*.
*   **Pitfalls:** Complex configuration/compatibility issues. Performance gain not guaranteed (may fallback to internal staging if RDMA unsupported). Harder debugging. Synchronization nuances still apply. Strong dependency on specific cluster environment. [5] **Note:** This would primarily optimize the halo exchange and Scatter/Gather steps in V4. It might not remove the need for D<->D copies or separate kernels if complex on-GPU padding/trimming logic isn't implemented. If trimming remains host-based, a final D->H copy is still needed.
*   **Table 2: MPI+CUDA Data Movement Strategies**
    | Strategy              | Description                                                              | Key Steps (Send Halo Example)                  | Pros                                   | Cons                                                 | Relevant Stage |
    | :-------------------- | :----------------------------------------------------------------------- | :--------------------------------------------- | :------------------------------------- | :--------------------------------------------------- | :------------- |
    | Manual Staging (V4)   | Explicit H<->D copies around MPI calls acting on host buffers            | `cudaMemcpy D2H(halo)` -> `MPI_Send(host_halo)` | Works anywhere, Explicit control       | High latency, PCIe load, More app code, Sync overhead | V4 (Current)   |
    | CUDA-aware (V5 Ideal) | Pass GPU ptr to MPI; Library uses optimized path (RDMA) for comms        | `MPI_Send(gpu_halo_ptr)`                       | Low latency(comms), Less PCIe(comms), Simpler comm code | Requires HW/SW support, Complex config/debug         | V5 (if works)  |
    | CUDA-aware (V5 Fback) | Pass GPU ptr; Library falls back to internal staging (mimics V4 staging) | `MPI_Send(gpu_halo_ptr)` -> Lib does copies    | Simpler app code (comms)               | Perf similar/worse than V4, Hides data path         | V5 (fallback)  |

---

## 7. HPC Software Engineering and Development Practices
*(Content updated for V4)*
*   **Code Organization:** V4 uses separate `.cpp`/`.cu` files but the `alexnetTileForwardCUDA` function encapsulates the entire GPU sequence, reducing modularity at the layer level within the hybrid context. Error handling uses `CUDA_CHECK` macro with `MPI_Abort`.
*   **Error Handling:** Essential for parallel debugging. Check return codes for *all* MPI and CUDA calls. Use `MPI_Error_string`, `cudaGetErrorString`. Use rank ID in MPI error messages. Use robust `CUDA_CHECK` macro that includes rank and aborts cleanly.
*   **Debugging Techniques:**
    *   *V1:* GDB, Valgrind.
    *   *V2:* Rank-based printf, parallel debuggers (multi-process GDB, TotalView, DDT), MPI analysis tools. Focus on communication issues (deadlocks, tags, sizes).
    *   *V3:* Kernel printf (use sparingly), `cuda-gdb`, `cuda-memcheck`/`compute-sanitizer`, strategic `cudaDeviceSynchronize`.
    *   *V4/V5:* Combine techniques. Use rank-based `fprintf(stderr,...)` for diagnostics. Parallel debuggers with CUDA awareness (TotalView, DDT) are ideal. Isolate issues (1 process vs multi-process vs multi-node). Use `cuda-memcheck mpirun ...` for NP=4 crash. Check MPI resource usage.
*   **Data Precision:** Use `float` (single-precision) for deep learning inference. GPUs are optimized for it [3]. `double` uses 2x memory/bandwidth and is much slower computationally. Ensure consistent use of `float` and `MPI_FLOAT`.

---

## 8. Performance Analysis, Bottlenecks, and Profiling

*   **Likely Bottlenecks per Stage:**
    *   *V1:* CPU compute, Host memory bandwidth.
    *   *V2:* MPI Comm (`Bcast`/`Gather`/Halo Exchange), Network B/W & Latency, Memory per node, Load imbalance, CPU compute.
    *   *V3:* GPU Kernel (compute/memory bound), **PCIe transfer time (H<->D)**, Kernel launch overhead.
    *   *V4 (Current):* **PCIe H<->D transfer of full padded tiles**, **Host staging/trimming CPU work**, MPI Halo Comm (Host), Sync overhead, Load imbalance, *Potentially internal inefficiencies in `alexnetTileForwardCUDA`*.
    *   *V5:* Network B/W & Latency (MPI comms potentially optimized), GPU Kernel, PCIe (if internal staging occurs or host logic remains), Sync overhead, Library efficiency, Load imbalance, Amdahl's Law.
*   **Standard Performance Metrics:** Wall Clock Time, Time Breakdowns (Compute vs Comm vs H2D/D2H vs Sync), Speedup (S(N) = T1/TN), Efficiency (E(N) = S(N)/N), Scalability (Strong/Weak), Communication Volume/Bandwidth, Kernel Execution Time, GPU Utilization/Occupancy/Memory Throughput.
*   **Suitable Profiling Tools/Techniques:**
    *   *Manual Timers:* `MPI_Wtime()`, CUDA Events (`cudaEventElapsedTime`). Crucial for V4 breakdown.
    *   *CPU Profilers:* `gprof`, `perf` (for host-side V4 logic).
    *   *MPI Profilers:* `mpiP`, Score-P, Vampir, TAU (analyze V2/V4 MPI phases).
    *   *CUDA Profilers:* Nsight Systems (`nsys` - system view, CPU/GPU/API timeline, **essential for V3/V4**), Nsight Compute (`ncu` - deep kernel analysis).
    *   *Combined:* Score-P, Vampir, TAU often handle both MPI+CUDA for unified view.
*   **Analysis Focus:** Decompose total time to identify true bottlenecks (esp. H<->D vs Compute vs Comm in V4). Understand how bottlenecks shift between stages. Compare against theoretical limits. Analyze numerical discrepancies.
*   **Table 3: Performance Metrics and Profiling Tools per Project Stage**
    | Stage            | Likely Bottlenecks                                                    | Key Metrics                                                    | Recommended Tools                                            |
    | :--------------- | :-------------------------------------------------------------------- | :------------------------------------------------------------- | :----------------------------------------------------------- |
    | V1 (Serial)      | CPU compute, Host memory                                              | Wall time, CPU cycles, Cache misses                            | `gprof`, `perf`, Manual timers                               |
    | V2 (MPI)         | MPI Comm (Halo/Bcast/Gather), Network, Mem/Node, CPU compute          | Wall time, Speedup/Efficiency, Time breakdown, MPI stats       | Manual `MPI_Wtime`, MPI Profilers (Score-P, Vampir, TAU)     |
    | V3 (CUDA)        | **PCIe transfer**, GPU kernel, Launch overhead                        | Wall time, Speedup(vsV1), Kernel/H2D/D2H times, GPU util/mem BW | Manual CUDA Events, Nsight Systems (`nsys`), Nsight Compute (`ncu`) |
    | V4 (MPI+CUDA)    | **PCIe H<->D (Tiles)**, MPI Comm (Host Halo), Host Trim, Sync, GPU Kernel | Wall time, Speedup/Eff(vsV1/V3), Breakdown, Net BW, Wait times | **Nsight Systems (`nsys`)**, Combined Profilers, Manual Timers, Debuggers (for crash) |
    | V5 (CUDA-aware)  | Network (Direct MPI), GPU Kernel, PCIe (if fallback/host logic), Lib eff | Wall time, Speedup/Eff(vsV4), Breakdown, Net BW (direct)      | Combined Profilers, `nsys`, Manual Timers, V4 Comparison      |

---

## 9. Advanced Topics, Libraries Context, and Potential Gaps

*   **Gaps in Current Plan:**
    *   *Async Operations/Overlap:* Critical technique (non-blocking MPI + CUDA streams + pinned memory) not implemented. Could significantly hide V4 latency.
    *   *Load Balancing:* Assumed balanced work; real scenarios often require balancing.
    *   *Advanced MPI:* Derived Datatypes, One-Sided Comm (RMA), advanced communicators not covered.
    *   *Topology Awareness:* Performance sensitive to network topology & process mapping, not addressed.
    *   *Alternative Paradigms:* OpenMP, Task-based (Legion), PGAS languages provide context.
*   **Awareness of Relevant Libraries:** Essential context for the manual implementation effort.
    *   *cuDNN:* NVIDIA's optimized library for DNN primitives (Conv, Pool, etc.). Performance benchmark. [8] Would replace manual kernels.
    *   *NCCL:* NVIDIA's optimized library for multi-GPU/multi-node collectives (Allreduce, Bcast). Standard for DL training. [7] Could replace manual MPI collectives if needed.
    *   *Thrust:* High-level C++ template library for CUDA (parallel algorithms, `device_vector`, pinned memory allocators). Could simplify memory management.
    *   *Vendor MPIs:* May offer better performance/CUDA integration than standard OpenMPI/MPICH on specific clusters.
*   **Validation and Numerical Correctness:** Need strategy to compare outputs between versions within a tolerance (epsilon) due to floating-point non-associativity. **Observed differences need investigation.**

---

## 10. Research-Based Recommendations for the Project

1.  **Standardize Environment:** Use target cluster or exact-replica containers (Docker/Singularity) for all testing/development, especially V4 debugging.
2.  **Prioritize V4 Debugging:** Fix output format, resolve NP=4 crash, **verify numerical correctness and host trimming logic.**
3.  **Profile V3 & V4:** Use `Nsight Systems` to understand V3 slowness and pinpoint V4 bottlenecks (H<->D vs Compute vs Host Comm vs Host Trim). Use `Nsight Compute` for kernel optimization if needed.
4.  **Mandate Pinned Memory:** Require use of `cudaMallocHost` or equivalents for H<->D transfer buffers if performance optimization (e.g., async overlap) is attempted later.
5.  **Support V4/V5 Integration & Analysis:** Provide guidance/templates for H<->D data flow, MPI/CUDA sync. **Verify V5 feasibility/benefit on target cluster before implementation.** Analyze the performance impact of the current V4 host-staging/tile approach vs alternatives.
6.  **Enforce SE Practices:** Mandate modularity, comprehensive error checking (`CUDA_CHECK` w/ `MPI_Abort`). Provide debugger/profiler access & training. Encourage use of version control.
7.  **Require Performance Analysis:** Mandate timing breakdowns, speedup/efficiency calculation, bottleneck analysis, and scalability discussion for V2-V4 (once working).
8.  **Address Numerical Discrepancies:** Investigate and explain (or fix) the observed differences in numerical output between versions.
9.  **Provide Context:** Discuss async overlap conceptually. Introduce cuDNN/NCCL as industry standards and performance references.

---

## 11. References
*(List unchanged)*
1.  Lawrence Mitchell, *“MPI: Domain decomposition and halo exchanges”*, Durham Univ. HPC Course ([Link](https://teaching.wence.uk/phys52015/exercises/mpi-stencil/))
2.  Wikipedia – *“Data parallelism vs. Model parallelism”* ([Link](https://en.wikipedia.org/wiki/Data_parallelism))
3.  NVIDIA Developer Blog – *“Defining Floating Point Precision (FP64, FP32, FP16)”*, Exxact Corp. (2024) ([Link](https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16))
4.  OpenUCX Documentation / CISL Tutorial Slides – Recommended environment settings for CUDA-aware MPI (UCX transport). ([Example Slide Link](https://www.cisl.ucar.edu/sites/default/files/2022-09/11_MultiGPU_Part2.slides%20%282%29.pdf))
5.  NVIDIA Forums – discussion *“Windows 11 + WSL + CUDA-aware MPI”* ([Link](https://forums.developer.nvidia.com/t/windows-11-wsl-cuda-aware-mpi-geforce-40-series-seg-fault-but-with-geforce-30-series-ok/292425))
6.  Alex Krizhevsky et al., *“ImageNet Classification with Deep CNNs (AlexNet)”*, NIPS 2012 ([Paper Link - Often Found Online](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))
7.  NVIDIA Developer – *“NCCL (Nvidia Collective Communication Library)”* ([Link](https://developer.nvidia.com/nccl))
8.  NVIDIA Documentation – *“NVIDIA cuDNN”* ([Link](https://docs.nvidia.com/deeplearning/cudnn/latest/))
9.  Stack Overflow – *“std::vector<T> contiguous memory”* ([Example Link](https://stackoverflow.com/questions/2009531/c-stdpair-stdvector-memcopy))
10. SCM Documentation – *“Known Issues: Intel MPI on WSL”* ([Link](https://www.scm.com/doc/Installation/Additional_Information_and_Known_Issues.html))

---

## 12. Rolling Research Notes & Findings

*(**Instructions:** Manually add your own specific findings, benchmark results, interesting articles, or unexpected behaviors encountered during development here.)*

*   **[Current Date]:** Confirmed V4 implements host-staging with full padded tile H<->D copy and GPU execution of entire sequence via `alexnetTileForwardCUDA`. Host trimming logic uses simple offsets.
*   **[Current Date]:** V4 output format mismatch identified as cause for parsing errors in `run_final_project.sh`.
*   **[Current Date]:** V4 NP=4 crash occurs (Exit 134), requires debugging via log file inspection, debuggers (`gdb`/`cuda-gdb`), or `cuda-memcheck`. Could be resource limit, memory error, or MPI bug.
*   **[Date]:** Note about specific MPI behavior observed on cluster vs WSL2...
*   **[Date]:** Benchmark results for V2.1 Broadcast All show communication time dominates beyond P=4...
*   **[Date]:** Found issue with LRN layer indexing in serial code, fixed in V1...
*   ... *(Add more entries as you progress)* ...

---