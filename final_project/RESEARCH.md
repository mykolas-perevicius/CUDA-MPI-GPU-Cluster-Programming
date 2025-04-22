# CS485 Final Project: Research, Analysis, and Findings

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation)

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
    *   [Comparison with Alternative CPU-MPI Schemes (e.g., V2.2 Scatter+Halo)](#42-comparison-with-alternative-cpu-mpi-schemes-eg-v22-scatterhalo)
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

The CS485 final project plan, implementing AlexNet inference via a five-stage parallelization (V1-V5), is pedagogically valuable for its incremental introduction to Serial, MPI, CUDA, and Hybrid MPI+CUDA programming. However, it presents significant technical and conceptual challenges. Key concerns identified through research include:
*   **Learning Curve:** Steep difficulty increases between stages (V1->V2, V3->V4).
*   **V2 MPI Strategy:** The initial "Broadcast All" (V2.1) approach is simple but scales poorly due to communication/memory overhead; alternatives like Scatter+Halo (V2.2) are more scalable but complex.
*   **Environment:** Developing in WSL2 introduces compatibility/performance risks compared to the target Fedora cluster.
*   **Data Structures:** `std::vector<float>` is suboptimal for CUDA host-device transfers; pinned memory is crucial for V3+.
*   **V5 Feasibility:** CUDA-aware MPI benefits depend heavily on specific cluster HW/SW support (GPU Direct RDMA).
*   **Bottlenecks:** Performance limitations shift dramatically (CPU -> MPI Comm -> GPU Kernel/Transfers -> MPI+CUDA Sync/Comm -> Network).
*   **Software Engineering:** Robust error handling, modularity, and effective debugging/profiling are critical but challenging.
*   **Context:** Plan lacks emphasis on asynchronous overlap and standard libraries (cuDNN, NCCL).

Recommendations focus on standardizing the environment, refining the V2 strategy (or its framing), mandating appropriate data structures by V3, providing strong support for V4 integration, verifying V5 viability, enforcing SE practices, and contextualizing manual efforts with industry libraries/techniques.

---

## 2. Pedagogical Evaluation of Staged (V1-V5) Approach

### 2.1 Analysis of the Learning Curve

The 5-stage approach offers a structured learning path:
*   **V1 (Serial):** Establishes functional correctness and a performance baseline. Reinforces algorithm understanding.
*   **V2 (MPI):** Introduces distributed memory concepts, explicit communication (`MPI_Bcast`, `MPI_Gather`), and process management. A major conceptual shift.
*   **V3 (CUDA):** Introduces GPU architecture, SIMT execution, kernel programming, and host-device memory management. Another distinct, significant conceptual leap.
*   **V4 (MPI+CUDA):** The core integration challenge, demanding orchestration across nodes and GPUs, managing data movement between host (MPI) and device (CUDA) memory spaces, and synchronization. Often the steepest curve. Performance dominated by inter-node communication.
*   **V5 (CUDA-aware MPI):** Optimization layer potentially simplifying V4 code and improving performance via direct GPU communication (e.g., GPU Direct RDMA). Requires specific HW/SW support; pedagogical value depends on demonstrable benefits.

**Overall:** The staging isolates concepts effectively, but instructors/students must anticipate difficulty spikes, especially V1->V2 and V3->V4.

### 2.2 Common Student Challenges

*   **Conceptual:** Distinguishing distributed vs. shared memory models. Understanding MPI communicators, CUDA execution/memory hierarchies, synchronization primitives (`MPI_Barrier`, `cudaDeviceSynchronize`, `__syncthreads`).
*   **Implementation:** Debugging parallel/distributed non-deterministic code. Managing complex C++/MPI/CUDA builds. Correct API usage and error checking. Performance tuning (identifying/addressing bottlenecks).
*   **Integration (V4/V5):** Sequencing MPI calls and CUDA operations correctly. Ensuring host/device data consistency. Avoiding deadlocks. Managing GPU affinity. Configuring/verifying CUDA-aware MPI.

### 2.3 Alignment with Course Objectives

This project aligns well with typical HPC course objectives:
*   Hands-on experience with parallelization concepts (data/task parallelism, speedup, efficiency).
*   MPI standard for distributed memory programming.
*   CUDA for GPU accelerator programming.
*   Hybrid MPI+CUDA models.
*   Performance analysis and profiling.
*   Application to a relevant domain (CNN inference).

---

## 3. Development Environment and Build System

### 3.1 WSL2 vs. Native Linux (Fedora): Compatibility and Performance

Using WSL2 (Ubuntu) for development while targeting Fedora 37 presents risks:
*   **WSL2 Overview:** Runs a real Linux kernel via lightweight VM. Good compatibility but not identical to native Linux.
*   **Performance Discrepancies:** Filesystem I/O (Windows<->WSL2) can be slow. Virtualized networking may differ from native MPI performance. GPU compute usually close, but potential overheads exist.
*   **Compatibility Risks:** Differences in kernel versions, system libraries (glibc), CUDA drivers (WSL vs. native), and MPI implementation behavior (e.g., Intel MPI unsupported [10], OpenMPI quirks [5]) can cause "works on my machine" issues. Consumer GPUs in WSL2 may lack features like GPUDirect RDMA [5].
*   **Mitigation:** Minimize environment differences. Use containers (Docker/Singularity) in both dev/target environments. Perform frequent testing on the actual Fedora cluster. Match toolchain versions (GCC 12, CUDA 12.x, Open MPI) precisely.

**Conclusion:** Sole reliance on WSL2 is risky. Debugging environment-specific issues is time-consuming. Prioritize testing on or containerizing the target environment.

### 3.2 Build System: Makefiles and Bash Scripting

*   **Suitability:** Makefiles + Bash are standard HPC tools, generally sufficient for this project scope. Manage compilation of C++/CUDA and linking against MPI/CUDA libraries.
*   **Complexity Management:** Requires well-organized Makefiles (variables, pattern rules, dependencies). Robust Bash scripts with error checking are needed for automation.
*   **Alternative (CMake):** Offers better cross-platform support, dependency finding (FindMPI, FindCUDA), build configurations, and test integration. Scales better for complex projects and is a valuable skill. Represents stronger software engineering practice, though with a slightly steeper initial learning curve than basic Makefiles.

---

## 4. Analysis of MPI Parallelization Strategies (V2 Focus)

### 4.1 Critique of "Broadcast All, Compute Slices" Approach (V2.1)

This is the initially chosen strategy for V2: `MPI_Bcast` full input/parameters, each rank computes an output slice independently, `MPI_Gather`/`MPI_Gatherv` final slices.
*   **Scalability Issues:**
    *   *Communication Bottleneck:* `MPI_Bcast` cost scales poorly with process count (P) and data size. Initial broadcast dominates time as P increases. Gather also adds overhead.
    *   *Memory Inefficiency:* Each rank stores full input/parameters, negating distributed memory benefits. Limited by single-node memory.
*   **Implementation Simplicity:** Main advantage. Ranks compute independently after broadcast, simplifying logic. Avoids complex halo exchanges.
*   **Pedagogical Value:** Introduces basic collectives but fails to teach scalable communication patterns (e.g., neighbor exchange) needed for efficient distributed applications. Poor foundation for V4/V5 performance. Makes overlapping communication/computation difficult.

### 4.2 Comparison with Alternative CPU-MPI Schemes (e.g., V2.2 Scatter+Halo)

*   **Input Spatial Decomposition (Scatter + Halo Exchange):**
    *   *Description:* Divide input map spatially among ranks. Exchange boundary data (halos) with neighbors (`MPI_Sendrecv`). Replicate parameters.
    *   *Pros:* Reduces activation memory per rank. Good load balancing potential. Neighbor communication can be more scalable than broadcast. Natural fit for convolution.
    *   *Cons:* Much higher implementation complexity (halo logic, boundary handling, potential deadlocks). [1]
*   **Filter Decomposition (Model Parallelism):**
    *   *Description:* Distribute filters among ranks. Replicate input. Combine partial outputs (`MPI_Allgather`/`MPI_Allreduce`).
    *   *Pros:* Reduces parameter memory per rank. Good for many filters.
    *   *Cons:* Replicates input activation memory. Significant communication for combining outputs. Often used with data parallelism.
*   **Table 1: Comparison of Strategies (V2 Context)**
    | Strategy Name             | Communication Pattern (Primary)            | Memory/Rank                               | Scalability Potential | Implementation Complexity | Suitability (Conv) | Suitability (FC) |
    | :------------------------ | :----------------------------------------- | :---------------------------------------- | :-------------------- | :------------------------ | :----------------- | :--------------- |
    | Broadcast All/Output Slice | `MPI_Bcast`, `MPI_Gather`/`Allgather`        | High (Full Input + Full Model)            | Poor                  | Low                       | Moderate           | Moderate         |
    | Input Spatial Decomp      | Neighbor `MPI_Sendrecv` (Halo)             | Low (Input Patch + Halos + Full Model)    | Good                  | High                      | High               | Low              |
    | Filter Decomp             | `MPI_Bcast` (Input), `Allgather`/`Allreduce` | Moderate (Full Input + Partial Model) | Moderate              | Moderate                  | Moderate           | High             |

### 4.3 Sub-Version Implementation Philosophy

Recognize that multiple valid strategies (like V2.1 vs V2.2) exist for each stage.
1.  **Identify & Document:** Note different approaches in `README.md` or `RESEARCH.md`.
2.  **Implement Primary:** Implement one chosen strategy first within the main version folder (e.g., V2.1 in `v2_mpi_only/`).
3.  **Explore Alternatives (Optional):** If time allows, implement alternatives in sub-folders (e.g., `v2_mpi_only/v2.2_scatter_halo/`) or via Git branches for comparison.

### 4.4 Correctness Pitfalls in MPI

*   **Collectives:** Ensuring all ranks participate, correct counts/displacements/datatypes, correct reduction operations. Off-by-one errors in slicing.
*   **Halo Exchange (if used):** Incorrect neighbor ranks, mismatched tags/counts, buffer errors, deadlocks (blocking send/recv order), boundary conditions.
*   **Data Types:** `MPI_Datatype` must match C++ memory layout precisely.

---

## 5. Layer Parallelization: AlexNet Subset & Challenges (V3 Focus)

### 5.1 Representativeness of Early AlexNet Blocks

*   **Layers Covered:** Conv1, ReLU1, Pool1, LRN1(?), Conv2, ReLU2, Pool2, LRN2.
*   **Characteristics:** Covers compute-intensive convolution, simple element-wise ReLU, local reduction pooling, and more complex cross-channel LRN. Representative of fundamental CNN operations.
*   **Limitations:** Omits deeper Conv layers (smaller spatial size, more channels) and Fully Connected layers (matrix multiplies, different bottlenecks). May give skewed perspective on optimal strategies for the whole network. Early layers have large spatial dimensions suitable for spatial decomposition; later layers might favor filter decomposition.

### 5.2 Inherent Difficulties in Layer Parallelization (MPI & CUDA)

*   **MPI:**
    *   *Convolution:* Data distribution needs halo exchange (if spatial decomp.) or input replication (if filter decomp.). Load balancing.
    *   *Pooling:* Simpler, but needs halos if spatial decomp. and window crosses boundary.
    *   *LRN:* Cross-channel/spatial dependencies make efficient distribution hard.
*   **CUDA:**
    *   *Convolution:* Efficient mapping of loops to threads/blocks/grids. Optimizing memory access (coalescing, shared memory tiling for reuse). Algorithms (im2col, Winograd, direct).
    *   *Pooling:* Simpler kernel, boundary checks, avoid shared memory bank conflicts if used.
    *   *LRN:* Efficient access to neighbors (spatial/channel). Shared memory useful but needs careful indexing.

### 5.3 Critique of `std::vector<float>` for HPC/CUDA

Using `std::vector<float>` (from V1) in V3+ presents performance issues:
*   **Memory Allocation:** Allocates standard pageable host memory.
*   **Host-Device Transfer:** `cudaMemcpy` is slower with pageable memory vs. pinned (page-locked) host memory (allocated via `cudaMallocHost`). Pinned memory enables asynchronous transfers (`cudaMemcpyAsync`) essential for overlapping communication/computation. `std::vector::data()` provides the pointer, but the memory type limits transfer speed and overlap capability. [9]
*   **Alignment:** Default alignment may not be optimal for SIMD/GPU coalescing.
*   **Recommendation:** Transition to pinned memory strategies by V3 for performance-critical data involved in transfers.
    *   *Alternatives:* Raw `float*` with `cudaMallocHost`/`cudaFreeHost`; `std::unique_ptr` with custom deleter for `cudaFreeHost`; `std::vector` with custom pinned allocator; Thrust library (`thrust::host_vector`, `thrust::device_vector`).

---

## 6. CUDA Implementation and MPI+CUDA Integration (V3-V5)

### 6.1 V3 (CUDA Kernels): Best Practices and Pitfalls

*   **Best Practices:** Map computation to CUDA hierarchy. Minimize H<->D transfers. Use pinned host memory. Optimize global memory access (coalescing). Use shared memory for data reuse (beware bank conflicts). Maximize arithmetic intensity. Avoid warp divergence. Use CUDA streams for overlap. Profile with Nsight Compute/Systems. Rigorous error checking (`cudaGetLastError`).
*   **Pitfalls:** Uncoalesced access. Global memory bottleneck. Shared memory bank conflicts. Low occupancy. Thread divergence. Forgetting error checks. Underestimating transfer overhead. Kernel indexing errors.

### 6.2 V4 (MPI+CUDA): Integration Challenges

*   **Host/Device Data Management:** Explicit staging via host memory is required for MPI calls with non-CUDA-aware MPI. `GPU -> cudaMemcpy D2H -> Host Buffer -> MPI_Send -> Network -> MPI_Recv -> Host Buffer -> cudaMemcpy H2D -> GPU`. This adds latency and PCIe load. Optimization involves minimizing/overlapping these steps.
*   **Synchronization:** Coordinate MPI calls (blocking/non-blocking) with asynchronous CUDA operations (kernels, async copies) using `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaEventRecord/Synchronize`. Incorrect sync leads to race conditions or deadlocks.
*   **GPU Affinity:** Ensure each MPI rank binds to a specific GPU on multi-GPU nodes (`cudaSetDevice` or `CUDA_VISIBLE_DEVICES`). Failure leads to resource contention and incorrect results.

### 6.3 V5 (CUDA-aware MPI): Requirements, Benefits, Pitfalls

*   **Requirements:** MPI library built with CUDA support (OpenMPI `--with-cuda`, MVAPICH2-GDR, etc.). Compatible CUDA Toolkit/drivers. HW support for GPU Direct RDMA (specific GPUs/NICs/fabric) for optimal performance. Correct system/MPI environment configuration (e.g., UCX settings [4, 5]).
*   **Benefits:** Pass GPU device pointers directly to MPI calls. Library handles transfer, potentially via RDMA bypassing host memory. Reduces latency, PCIe traffic, CPU overhead. Simplifies application code (removes manual `cudaMemcpy` staging).
*   **Pitfalls:** Complex configuration/compatibility issues. Performance gain not guaranteed (may fallback to internal staging if RDMA unsupported). Harder debugging. Synchronization nuances still apply. Strong dependency on specific cluster environment. [5]
*   **Table 2: MPI+CUDA Data Movement Strategies**
    | Strategy              | Description                                                              | Key Steps (Send)                               | Pros                                   | Cons                                          | Relevant Stage |
    | :-------------------- | :----------------------------------------------------------------------- | :--------------------------------------------- | :------------------------------------- | :-------------------------------------------- | :------------- |
    | Manual Staging (V4)   | Explicit H<->D copies around MPI calls                                   | Kernel -> `cudaMemcpy D2H` -> `MPI_Send`       | Works anywhere, Explicit control       | High latency, PCIe load, More app code        | V4             |
    | CUDA-aware (V5 Ideal) | Pass GPU ptr to MPI; Library uses optimized path (RDMA)                  | Kernel -> `MPI_Send(gpu_ptr)`                  | Low latency, Less PCIe load, Simpler code | Requires HW/SW support, Complex config/debug | V5 (if works)  |
    | CUDA-aware (V5 Fback) | Pass GPU ptr; Library falls back to internal staging (mimics V4)         | Kernel -> `MPI_Send(gpu_ptr)` -> Lib does copies | Simpler app code                       | Perf similar/worse than V4, Hides data path | V5 (fallback)  |

---

## 7. HPC Software Engineering and Development Practices

*   **Code Organization:** Crucial for managing complexity. Use separation of concerns (layers vs MPI vs CUDA logic), clear directory structure (`src`, `include`, `kernels`), encapsulation (namespaces, classes, functions).
*   **Error Handling:** Essential for parallel debugging. Check return codes for *all* MPI and CUDA calls. Use `MPI_Error_string`, `cudaGetErrorString`. Use rank ID in MPI error messages. Consider helper macros for CUDA error checking.
*   **Debugging Techniques:**
    *   *V1:* GDB, Valgrind.
    *   *V2:* Rank-based printf, parallel debuggers (multi-process GDB, TotalView, DDT), MPI analysis tools (if available). Focus on communication issues (deadlocks, tags, sizes).
    *   *V3:* Kernel printf (use sparingly), `cuda-gdb`, `cuda-memcheck`/`compute-sanitizer`, strategic `cudaDeviceSynchronize`.
    *   *V4/V5:* Combine techniques. Parallel debuggers with CUDA awareness (TotalView, DDT) are ideal. Isolate issues (1 process vs multi-process vs multi-node).
*   **Data Precision:** Use `float` (single-precision) for deep learning inference. GPUs are optimized for it [3]. `double` uses 2x memory/bandwidth and is much slower computationally. Ensure consistent use of `float` and `MPI_FLOAT`.

---

## 8. Performance Analysis, Bottlenecks, and Profiling

*   **Likely Bottlenecks per Stage:**
    *   *V1:* CPU compute, Host memory.
    *   *V2:* `MPI_Bcast` / `MPI_Gather` time, Network B/W & Latency, Memory per node.
    *   *V3:* GPU Kernel (compute/memory bound), PCIe transfer time.
    *   *V4:* Inter-node MPI (host staging), Sync overhead, PCIe contention, Load imbalance.
    *   *V5:* Inter-node MPI (direct path), Network B/W & Latency, Library efficiency, Load imbalance, Amdahl's Law.
*   **Standard Performance Metrics:** Wall Clock Time, Time Breakdowns (Compute vs Comm vs H2D/D2H vs Sync), Speedup (S(N) = T1/TN), Efficiency (E(N) = S(N)/N), Scalability (Strong/Weak), Communication Volume/Bandwidth, Kernel Execution Time, GPU Utilization/Occupancy/Memory Throughput.
*   **Suitable Profiling Tools/Techniques:**
    *   *Manual Timers:* `MPI_Wtime()`, CUDA Events (`cudaEventElapsedTime`).
    *   *CPU Profilers:* `gprof`, `perf`.
    *   *MPI Profilers:* `mpiP`, Score-P, Vampir, TAU (provide timelines, message stats, wait state analysis).
    *   *CUDA Profilers:* Nsight Systems (`nsys` - system view, CPU/GPU/API timeline), Nsight Compute (`ncu` - deep kernel analysis, perf counters, bottlenecks).
    *   *Combined:* Score-P, Vampir, TAU often handle both MPI+CUDA for unified view.
*   **Analysis Focus:** Decompose total time to identify true bottlenecks. Understand how bottlenecks shift between stages. Compare against theoretical limits (Amdahl's Law).
*   **Table 3: Performance Metrics and Profiling Tools per Project Stage**
    | Stage            | Likely Bottlenecks                                   | Key Metrics                                                    | Recommended Tools                                            |
    | :--------------- | :--------------------------------------------------- | :------------------------------------------------------------- | :----------------------------------------------------------- |
    | V1 (Serial)      | CPU compute, Host memory                             | Wall time, CPU cycles, Cache misses                            | `gprof`, `perf`, Manual timers                               |
    | V2 (MPI)         | `MPI_Bcast`/`Gather`, Network, Mem/Node, CPU compute | Wall time, Speedup/Efficiency, Time breakdown, MPI stats       | Manual `MPI_Wtime`, MPI Profilers (Score-P, Vampir, TAU)     |
    | V3 (CUDA)        | GPU kernel, PCIe transfer, Launch overhead           | Wall time, Speedup(vsV1), Kernel/H2D/D2H times, GPU util/mem BW | Manual CUDA Events, Nsight Systems (`nsys`), Nsight Compute (`ncu`) |
    | V4 (MPI+CUDA)    | MPI Comm (Host staging), Sync, Load imbalance        | Wall time, Speedup/Eff(vsV1/V3), Breakdown, Net BW, Wait times | Combined Profilers (Score-P, Vampir, TAU), `nsys`, Manual Timers |
    | V5 (CUDA-aware)  | MPI Comm (Direct), Network, Lib efficiency, Amdahl   | Wall time, Speedup/Eff(vsV4), Breakdown, Net BW (direct)      | Combined Profilers, `nsys`, Manual Timers, V4 Comparison      |

---

## 9. Advanced Topics, Libraries Context, and Potential Gaps

*   **Gaps in Current Plan:**
    *   *Async Operations/Overlap:* Critical technique (non-blocking MPI + CUDA streams) not explicitly planned. Should be discussed/introduced conceptually.
    *   *Load Balancing:* Assumed balanced work; real scenarios often require explicit balancing strategies.
    *   *Advanced MPI:* Derived Datatypes, One-Sided Comm (RMA), advanced communicators not covered.
    *   *Topology Awareness:* Performance sensitive to network topology & process mapping, not addressed.
    *   *Alternative Paradigms:* OpenMP, Task-based (Legion), PGAS languages provide context.
*   **Awareness of Relevant Libraries:** Essential context for the manual implementation effort.
    *   *cuDNN:* NVIDIA's optimized library for DNN primitives (Conv, Pool, etc.). Performance benchmark. [8]
    *   *NCCL:* NVIDIA's optimized library for multi-GPU/multi-node collectives (Allreduce, Bcast). Standard for DL training. [7]
    *   *Thrust:* High-level C++ template library for CUDA (parallel algorithms, `device_vector`).
    *   *Vendor MPIs:* May offer better performance/CUDA integration than standard OpenMPI/MPICH on specific clusters.
*   **Validation and Numerical Correctness:** Need strategy to compare outputs between versions within a tolerance (epsilon) due to floating-point non-associativity.

---

## 10. Research-Based Recommendations for the Project

1.  **Standardize Environment:** Use target cluster or exact-replica containers (Docker/Singularity) for all testing/development.
2.  **Refine V2 Strategy:** Either replace "Broadcast All" with Spatial Decomposition (teaching halo exchange) OR clearly document limitations of Broadcast All and frame as introductory step.
3.  **Mandate Pinned Memory by V3:** Require use of `cudaMallocHost` or equivalents for H<->D transfer buffers to enable `cudaMemcpyAsync`.
4.  **Focus V3 Pedagogy:** Emphasize CUDA fundamentals (coalescing, shared mem, sync) and profiling (Nsight), not matching cuDNN performance.
5.  **Support V4/V5 Integration:** Provide guidance/templates for H<->D data flow, MPI/CUDA sync. Verify V5 feasibility/benefit on target cluster before assignment.
6.  **Enforce SE Practices:** Mandate modularity, comprehensive error checking from V1. Provide debugger/profiler access & training.
7.  **Require Performance Analysis:** Mandate timing breakdowns, speedup/efficiency calculation, bottleneck analysis, and scalability discussion for V2-V5.
8.  **Provide Context:** Discuss async overlap conceptually. Introduce cuDNN/NCCL as industry standards and performance references.

---

## 11. References

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

*   **YYYY-MM-DD:** Note about specific MPI behavior observed on cluster vs WSL2...
*   **YYYY-MM-DD:** Benchmark results for V2.1 Broadcast All show communication time dominates beyond P=4...
*   **YYYY-MM-DD:** Found issue with LRN layer indexing in serial code, fixed in V1...
*   ... *(Add more entries as you progress)* ...

---