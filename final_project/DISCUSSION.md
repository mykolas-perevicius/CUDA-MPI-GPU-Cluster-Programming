# CS485 Final Project: Discussion Log & Professor Meeting Prep

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation) - Blocks 1 & 2 Focus
**Student:** [Your Name]
**Date Started:** [Date file created]
**Last Updated:** [Current Date]

**Purpose:** This document serves as a rolling log to prepare for discussions with Professor Sohn regarding the CS485 final project. It outlines current status, challenges, questions, and proposed next steps. Meeting summaries will be manually added at the bottom after each discussion.

---

## 1. Project Overview & Goal Reminder

*   **Objective:** Implement and benchmark inference for **AlexNet Blocks 1 & 2** using a 5-stage approach: V1 (Serial), V2 (MPI CPU), V3 (CUDA GPU), V4 (MPI+CUDA), V5 (CUDA-Aware MPI). The primary goal is learning parallelization techniques and performance analysis on this subset.
*   **Target Environment:** Fedora 37, GCC 12, CUDA 12.x, Open MPI.
*   **Deliverable:** Working code for completed versions (up to V5 ideally), focusing on a comparative performance analysis across the stages. Presentation will highlight the parallelization journey.

---

## 2. Current Status & Immediate Next Step

*   **Last Completed Stages:**
    *   V1 (Serial CPU) - **Completed & Validated.**
    *   V2 (MPI Only - Approach 2.1: Broadcast All) - **Completed.** Demonstrated poor scaling.
    *   V2 (MPI Only - Approach 2.2: Scatter+Halo) - **Completed.** Demonstrated expected speedup.
    *   V3 (CUDA Only - Single GPU) - **Completed.** Functional, needs profiling/optimization.
*   **Current Focus:** **Version 4 (MPI + CUDA Hybrid)**
*   **Working Directory:** `final_project/v4_mpi_cuda/`
*   **Immediate Goal:** Integrate the V2.2 (Scatter+Halo) MPI logic with the V3 CUDA kernels. This involves managing data distribution via MPI on host buffers, implementing explicit Host<->Device copies (`cudaMemcpy`) around MPI communication and kernel launches, ensuring GPU affinity, and synchronizing MPI/CUDA operations. Build using `nvcc -ccbin=mpicxx`.

---

## 3. Recent Accomplishments / Milestones Achieved

*   Completed V1 (Serial) implementation and established baseline performance (~617ms).
*   Implemented V2 MPI using two distinct strategies:
    *   V2.1 (Broadcast All): Validated basic MPI, highlighted scalability issues (~660ms -> ~802ms for np=1->4).
    *   V2.2 (Scatter+Halo): Successfully implemented more complex halo exchange logic using non-blocking MPI, demonstrating expected speedup (~491ms -> ~177ms for np=1->4).
*   Completed V3 (CUDA Only) implementation: Ported layer logic to basic CUDA kernels, functional on single GPU (~750ms, needs optimization/profiling).
*   Established robust project structure with versioned directories and comprehensive documentation (`README.md`, `RESEARCH.md`, `ai_context.txt`, `discussion.md`).

---

## 4. Current Challenges / Roadblocks / Issues

*   **Anticipated for V4:**
    *   **MPI+CUDA Integration Complexity:** Correctly orchestrating the sequence: `MPI_Recv (Host)` -> `cudaMemcpy H2D` -> `Launch Kernel` -> `cudaDeviceSync/EventSync` -> `cudaMemcpy D2H` -> `MPI_Send (Host)`. This applies particularly to halo exchanges.
    *   **Synchronization:** Ensuring CUDA operations complete before data is used by MPI (and vice-versa) without introducing unnecessary serialization or deadlocks. Correct use of `MPI_Wait/Test` and CUDA Events/Streams.
    *   **Host Staging Bottleneck:** The explicit H<->D copies required before/after MPI calls are expected to be a significant performance bottleneck compared to pure CUDA or potential V5. Quantifying this overhead will be key.
    *   **GPU Affinity:** Correctly setting `cudaSetDevice` based on local rank to ensure each MPI process uses its designated GPU on multi-GPU nodes.
    *   **Debugging Hybrid Code:** Debugging issues that might stem from the interaction between MPI state and CUDA state will be challenging. Requires combined debugging approaches.
    *   **Makefile Complexity:** Ensuring the `nvcc -ccbin=mpicxx` build process correctly compiles CUDA (`.cu`) and C++ (`.cpp`) files and links against both CUDA and MPI libraries.

---

## 5. Specific Questions for Professor Sohn

*(**Instructions:** Review/update before meeting)*

1.  **V4 Integration Strategy:** Given V2.2 (Scatter+Halo) and V3 (CUDA kernels) are complete, what are common pitfalls or recommended patterns for structuring the V4 hybrid code, specifically regarding the placement and synchronization of `cudaMemcpy` calls around the MPI halo exchange logic (`MPI_Isend`/`Irecv`/`Wait`)?
2.  **V4 Performance Expectation:** Considering the host staging overhead, is it expected that V4 (MPI+CUDA) might initially perform *worse* than V3 (CUDA only) or even V2.2 (MPI only) for a small number of processes/nodes? How should we interpret such results?
3.  **Asynchronous Operations (Overlap):** Should we attempt to implement asynchronous overlap (CUDA streams with `cudaMemcpyAsync` and non-blocking MPI) in V4/V5, or is mastering the synchronous host-staging approach sufficient for V4?
4.  **V5 Feasibility Check:** What steps should we take to verify if the target cluster's Open MPI installation is truly CUDA-aware and supports efficient GPU Direct RDMA? Are there specific environment variables or test programs recommended?
5.  **Profiling Hybrid Code:** What's the recommended approach for profiling V4/V5 on the cluster? Can Nsight Systems effectively capture both MPI and CUDA timelines, or should we primarily rely on manual `MPI_Wtime` and CUDA Event timers?
6.  **Presentation Scope Confirmation:** Reconfirming that the primary presentation deliverable is the comparative analysis of V1-V5 performance for Blocks 1&2, rather than a fully implemented AlexNet.

---

## 6. Proposed Next Steps / Plan

*(**Instructions:** Outline plan until next meeting)*

1.  **Implement V4 (`v4_mpi_cuda/`):**
    *   Start with restored baseline code.
    *   Integrate V2.2 `main.cpp` MPI logic (Scatterv, Isend/Irecv/Wait for halo, Gatherv).
    *   Integrate V3 `alexnet_cuda.cu` kernel launch logic.
    *   Implement **explicit H<->D copies** (`cudaMemcpy`) around MPI calls (halo exchange, initial scatter, final gather).
    *   Copy V3 CUDA kernels (`layers_cuda.cu`) and headers.
    *   Implement GPU affinity (`cudaSetDevice`).
    *   Implement necessary synchronization (`MPI_Wait`, `cudaDeviceSynchronize` or Events).
    *   Update `Makefile` for `nvcc -ccbin=mpicxx`.
2.  **Compile & Test V4:** Use `make` and `mpirun -np [2, 4, 8] ./template` (on appropriate nodes/GPUs).
3.  **Debug V4:** Resolve integration issues, focusing on data consistency between host/device and correct synchronization.
4.  **Initial V4 Performance Measurement:** Record wall time using `MPI_Wtime` and CUDA Events. Compare rough timing with V2.2 and V3.
5.  **Plan V5:** Based on V4 experience and cluster verification, outline specific code changes needed for CUDA-aware MPI calls.

---

## 7. Design Decisions & Considerations

*(**Instructions:** Note significant choices made and alternatives considered)*

*   **V2 Strategy Choice:** Implemented both V2.1 (Broadcast) and V2.2 (Scatter+Halo) to directly compare simple vs. scalable MPI approaches. V2.2 provides the better foundation for V4.
*   **V3 Kernels:** Implemented basic, functionally correct CUDA kernels. Performance optimization deferred to later or as part of V4/V5 profiling.
*   **V4 Initial Plan:** Focus on correct synchronous host-staging implementation first before attempting asynchronous overlap.
*   **Data Structures:** Using `std::vector` on host, raw `float*` (`cudaMalloc`) on device for V3+. Pinned host memory (`cudaMallocHost`) should be considered for V4/V5 staging buffers to enable async copies later.

---

## 8. Performance & Benchmarking (WSL2 Dev Machine - Blocks 1&2)

*   **V1 (Serial):** ~617 ms (np=1)
*   **V2 (MPI - 2.1 Broadcast):** ~660ms (np=1), ~704ms (np=2), ~802ms (np=4) -> **Degrades**
*   **V2 (MPI - 2.2 Scatter+Halo):** ~491ms (np=1), ~334ms (np=2), ~177ms (np=4) -> **Speeds up**
*   **V3 (CUDA):** ~750 ms (np=1) -> **Needs Profiling/Optimization**
*   **V4 (MPI+CUDA):** *(To be added)*
*   **V5 (CUDA-Aware MPI):** *(To be added)*

*(Note: Absolute times will differ on target cluster. Relative scaling and bottlenecks are key points of analysis.)*

---
---

## Meeting Summaries & Action Items

*(**Instructions:** Manually add notes after each meeting with the professor)*

**YYYY-MM-DD - Discussion with Prof. Sohn**
*   **Topics Discussed:** ...
*   **Key Feedback / Decisions:** ...
*   **Action Items:** ...

---
