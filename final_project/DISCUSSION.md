# CS485 Final Project: Discussion Log & Professor Meeting Prep

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation) - Blocks 1 & 2 Focus
**Student:** Mykhaylo Kopylov
**Date Started:** [Date file created]
**Last Updated:** [Current Date]

**Purpose:** This document serves as a rolling log to prepare for discussions with Professor Sohn regarding the CS485 final project. The top section provides a narrative script for a meeting update, followed by detailed sections covering status, challenges, questions, and next steps. Meeting summaries will be manually added at the bottom after each discussion.

---

## Professor Meeting Opening Script/Narrative (Approx. 5-10 min verbal walkthrough)

"Hi Professor, thanks for meeting. I wanted to give you an update on the CS485 final project – the staged MPI+CUDA implementation of the first two blocks of AlexNet inference.

Just to recap, the main goal is to implement and benchmark these initial layers (Conv1 through LRN2) across five stages: starting with a basic serial version, then adding MPI parallelism on the CPU, then a CUDA-only version on the GPU, followed by a hybrid MPI+CUDA version, and finally an optional optimization using CUDA-aware MPI. The focus is really on learning the parallelization techniques and analyzing the performance trade-offs at each stage using this consistent workload.

So far, the first three versions are functionally complete.

*   **V1 (Serial)** is working correctly and gives us a baseline performance measurement. On my development machine, the test script clocks it around 600-700 milliseconds.
*   For **V2 (MPI Only)**, I actually implemented two different approaches as planned in the `RESEARCH.md` to see the contrast.
    *   `V2.1` used a simple 'broadcast all' strategy. As expected, it ran correctly but showed negative scaling – it got slower as we added more processes, likely due to the broadcast overhead and redundant computations.
    *   `V2.2` used a more standard 'scatter with halo exchange' approach, using non-blocking MPI calls. This was more complex to implement, especially handling the halo padding and the necessary trimming after the pooling layers, but it showed good strong scaling, getting significantly faster with 2 and 4 processes compared to V1 or V2.1. This confirms the value of distributing the data properly.
*   **V3 (CUDA Only)** ports the layer logic to CUDA kernels running on a single GPU. This version also runs to completion. However, the performance is currently quite slow – significantly slower than the V1 serial version, actually. This strongly suggests there are bottlenecks, likely either in the host-to-device data transfers or perhaps the kernels themselves aren't very optimized yet. This is something that definitely needs profiling with Nsight tools.
*   An important observation across V1, V2, and V3 is that the sample output values reported by the test script seem to differ quite a bit between the versions, and even between different process counts in V2.2. This is something I need to investigate further – it could be minor floating-point differences, or potentially initialization variations or even subtle bugs.

Now, that brings us to the current focus: **Version 4 (MPI + CUDA Hybrid)**.

*   The goal here is to combine the scalable MPI structure from V2.2 (scatter, host halo exchange, gather) with the GPU computation from V3.
*   I've implemented a version of V4 based on a host-staging strategy. The overall flow is: MPI scatters the input rows to host buffers on each rank, then the ranks perform the first halo exchange using MPI on these host buffers. The resulting padded data tile (local rows plus halos) is then copied from the host to the GPU. A helper function then takes over on the GPU and runs the *entire* sequence of layers – Conv1, ReLU1, Pool1, Conv2, ReLU2, Pool2, LRN2 – all on that single tile. The final result tile is copied back from the GPU to the host, and then the rows corresponding to the halo contribution are trimmed off on the host before the final MPI gather. Each rank also sets its GPU using `cudaSetDevice`.
*   **The status is that the V4 code is implemented, but it's currently in the debugging phase.** The automated test script (`run_final_project.sh`) shows it runs successfully for 1 and 2 processes, but it flags warnings because the output format (`shape...` / `sample...`) in my code doesn't exactly match what the script is trying to parse (`Final Output Shape...` / `Final Output (first 10 values)...`). That's a straightforward fix I need to make in the `cout` statements.
*   More importantly, **when run with 4 processes (NP=4), V4 crashes** with an MPI error (exit code 134). I need to dig into the log file for that run and likely use debuggers or `cuda-memcheck` to figure out if it's a resource limit issue, a memory access error, or perhaps a subtle bug in the MPI communication logic (like buffer sizes or offsets) that only manifests at NP=4.
*   Finally, once it's running correctly, I also need to **verify the numerical output** against V1/V3 and double-check that the host-side trimming logic is correctly removing the halo influence for all ranks and process counts.

So, the **immediate next steps** are squarely focused on debugging V4: fixing the output format, resolving the NP=4 crash, and verifying the correctness. Once V4 is stable and working correctly, the plan is to profile it and V3 properly, analyze the performance, and then evaluate the feasibility and potential benefit of implementing V5 (CUDA-aware MPI) based on the cluster's capabilities.

With that overview, I can show you the output from the `run_final_project.sh` script which summarizes the status of V1 through V4, including the warnings and the crash indication for V4..."

**(End of Script - Transition to showing results/demo/addressing specific questions)**

---
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
    *   V3 (CUDA Only - Single GPU) - **Completed.** Functional, needs profiling/optimization; numerical output differs from V1/V2.
*   **Current Focus:** **Version 4 (MPI + CUDA Hybrid) - Implemented, Currently Debugging.**
*   **Working Directory:** `final_project/v4_mpi_cuda/`
*   **Immediate Goal:** **Debug V4 implementation:**
    1.  Fix output format in `main_mpi_cuda.cpp` to match test script expectations.
    2.  Diagnose and resolve the runtime crash occurring with NP=4 (Exit Code 134).
    3.  Verify numerical correctness of V4 output and the host-side trimming logic against V1/V3 results.

---

## 3. Recent Accomplishments / Milestones Achieved

*   Completed V1 (Serial) implementation and established baseline performance (~667ms via script).
*   Implemented V2 MPI using two distinct strategies:
    *   V2.1 (Broadcast All): Validated basic MPI, highlighted scalability issues (~679ms -> ~881ms for np=1->4).
    *   V2.2 (Scatter+Halo): Successfully implemented halo exchange logic using non-blocking MPI, demonstrating expected speedup (~561ms -> ~281ms for np=1->4).
*   Completed V3 (CUDA Only) implementation: Ported layer logic to basic CUDA kernels, functional on single GPU (~2349ms, significantly slower than V1, needs profiling). Noted numerical differences.
*   **Implemented V4 (MPI+CUDA Host-Staging approach)** using V2.2 MPI structure and V3 CUDA kernels via `alexnetTileForwardCUDA` helper (currently requires debugging).
*   Established robust project structure with versioned directories and comprehensive documentation (`README.md`, `RESEARCH.md`, `AI.md`, `discussion.md`). V4 includes `bear`/`clang-tidy` integration.

---

## 4. Current Challenges / Roadblocks / Issues

*   **Specific V4 Debugging Needs:**
    *   **Output Format Mismatch:** `cout` statements in `main_mpi_cuda.cpp` differ from `run_final_project.sh` parser expectations (`shape`/`sample` vs `Final Output Shape`/`Final Output (first 10 values)`), preventing automated results capture. (Requires simple code change).
    *   **Runtime Crash (NP=4):** V4 fails with MPI Exit Code 134 when run with 4 processes. Requires investigation using logs (`logs/final_project_v4_np4.log`), debuggers (`gdb`, `cuda-gdb`), or memory checkers (`cuda-memcheck`). Could be resource limits, memory error (host/device), or MPI bug (e.g., incorrect buffer size/offset calculation in comms at NP=4).
    *   **Correctness Verification:** Need to confirm numerical output of V4 matches V1/V3 (within tolerance). The host-side trimming logic applied after the D->H copy needs careful validation for all ranks and NP values.
*   **V3 Performance:** V3 (CUDA Only) is much slower than V1 (Serial), indicating likely bottlenecks in H<->D transfers or kernel execution that need profiling with Nsight tools.
*   **Numerical Differences:** Sample outputs vary significantly across V1, V2.1, V2.2(NP>1), and V3. Needs investigation (initialization differences? floating-point non-associativity? bugs?).
*   **MPI+CUDA Integration Complexity:** Even with V4 implemented, ensuring correct synchronization, data flow (esp. padding/trimming), and resource management between MPI and CUDA remains inherently complex. The current V4 strategy (full tile H<->D, internal GPU sequence, host trim) may have performance implications to analyze later.

---

## 5. Specific Questions for Professor Sohn

*(**Instructions:** Review/update before meeting)*

1.  **V4 Debugging Guidance:** Regarding the V4 NP=4 crash (Exit 134), are there common MPI resource limits or configuration issues on the target cluster we should check first? Any advice on effectively debugging hybrid MPI+CUDA code, particularly potential interactions leading to such errors? Also, any tips for validating the host-side output trimming logic?
2.  **V4 Performance Expectation:** Once V4 is working, considering the current host-staging overhead (full tile H<->D copies), is it still expected that it might perform *worse* than V3 (CUDA only) or even V2.2 (MPI only) for a small number of processes/nodes? How should we interpret such results if they occur?
3.  **Asynchronous Operations (Overlap):** *After* V4 is debugged and profiled, would attempting asynchronous overlap (CUDA streams with `cudaMemcpyAsync` on pinned memory, non-blocking MPI) be a valuable optimization step for this project, or is mastering the synchronous host-staging approach sufficient?
4.  **V5 Feasibility Check:** What steps should we take to verify if the target cluster's Open MPI installation is truly CUDA-aware and supports efficient GPU Direct RDMA? Are there specific environment variables (e.g., UCX) or test programs recommended?
5.  **Profiling Hybrid Code:** What's the recommended approach for profiling V4/V5 on the cluster? Can Nsight Systems effectively capture both MPI wait times and CUDA kernel/memcpy timelines, or should we primarily rely on manual `MPI_Wtime` and CUDA Event timers inserted into the code?
6.  **Presentation Scope Confirmation:** Reconfirming that the primary presentation deliverable is the comparative analysis of V1-V4/V5 performance for Blocks 1&2, including the debugging journey, rather than a fully implemented AlexNet.
7.  **Numerical Differences:** We've observed different sample output values across V1, V2 (different strategies/NPs), and V3. How should we approach investigating these? Is minor variation expected due to floating-point math, or does it likely indicate bugs or initialization differences?

---

## 6. Proposed Next Steps / Plan

*(**Instructions:** Outline plan until next meeting)*

1.  **Debug V4 (`v4_mpi_cuda/`):**
    *   **Fix Output Format:** Modify `cout` lines in `main_mpi_cuda.cpp` to match `run_final_project.sh` expectations. Re-run script to confirm parsing success for NP=1, 2.
    *   **Investigate NP=4 Crash:** Analyze log file (`logs/final_project_v4_np4.log`). Run manually with debuggers/memcheck as needed to identify root cause (memory, MPI comms, resource limits?). Implement fix.
    *   **Verify Correctness:** Compare V4 numerical output (NP=1, 2, 4 once working) against V1/V3 reference. Review and test the host-side trimming logic for correctness across different ranks/NPs.
2.  **Profile V3:** Use Nsight Systems/Compute to understand why V3 is slow compared to V1. Identify kernel vs. H<->D bottlenecks.
3.  **Initial V4 Performance Measurement:** Once V4 is working correctly for NP=1, 2, 4, record wall times using `MPI_Wtime` and/or CUDA Events. Compare rough timing with V2.2 and V3.
4.  **Profile V4:** Use Nsight Systems to analyze the execution flow and identify key bottlenecks (H<->D copies, host halo exchange, host trimming, GPU compute time).
5.  **Plan V5 / Optimizations:** Based on V4 debugging/profiling experience and cluster verification (Question 4), decide on feasibility/approach for V5 (CUDA-Aware MPI) or other optimizations (e.g., async overlap).

---

## 7. Design Decisions & Considerations

*(**Instructions:** Note significant choices made and alternatives considered)*

*   **V2 Strategy Choice:** Implemented both V2.1 (Broadcast) and V2.2 (Scatter+Halo) to directly compare simple vs. scalable MPI approaches. V2.2 provides the better foundation for V4.
*   **V3 Kernels:** Implemented basic, functionally correct CUDA kernels. Performance optimization deferred. Slowness relative to V1 is noted.
*   **V4 Implementation Choice:** Current V4 uses a host-staging approach where the entire padded tile is copied H<->D, and a helper function (`alexnetTileForwardCUDA`) runs the full GPU sequence internally. Trimming is done on the host after D->H copy. This simplifies the main MPI loop but might introduce H<->D bottlenecks and hides layer-level GPU details within the helper. An alternative function (`alexnetForwardPassMPI_CUDA`) exists but is currently unused.
*   **Data Structures:** Using `std::vector` on host, raw `float*` (`cudaMalloc`) on device for V3/V4. Pinned host memory (`cudaMallocHost`) should be considered for V4/V5 staging buffers if async copies are attempted.

---

## 8. Performance & Benchmarking (WSL2 Dev Machine - Script Output)

| Version                | Procs | Shape     | Time       | Status | Notes                                      |
| :--------------------- | :---- | :-------- | :--------- | :----- | :----------------------------------------- |
| V1 Serial              | 1     | 13x13x256 | ~667 ms    | ✔      | Baseline                                   |
| V2 2.1-broadcast-all | 4     | 13x13x256 | ~881 ms    | ✔      | Degrades (Expected)                        |
| V2 2.2-scatter-halo  | 4     | 13x13x256 | ~281 ms    | ✔      | Scales Well (Expected)                     |
| V3 CUDA                | 1     | 13x13x256 | ~2349 ms   | ✔      | Slow (Needs Profiling)                     |
| V4 MPI+CUDA          | 1     | –         | –          | ⚠      | Runs; Output format issue                  |
| V4 MPI+CUDA          | 2     | –         | –          | ⚠      | Runs; Output format issue                  |
| V4 MPI+CUDA          | 4     | –         | –          | ⚠      | CRASH (Exit 134)                           |

*(Note: Absolute times will differ on target cluster. Relative scaling and bottlenecks are key points of analysis. V4 status reflects known issues.)*

---
---

## Meeting Summaries & Action Items

*(**Instructions:** Manually add notes after each meeting with the professor)*

**YYYY-MM-DD - Discussion with Prof. Sohn**
*   **Topics Discussed:** ...
*   **Key Feedback / Decisions:** ...
*   **Action Items:** ...

---