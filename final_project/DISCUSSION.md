# Project Discussion

**Project:** AlexNet Inference (MPI+CUDA Staged Implementation)
**Purpose:** This document serves as a rolling log to prepare for discussions with Professor Sohn regarding the CS485 final project. It outlines current status, challenges, questions, and proposed next steps. Meeting summaries will be manually added at the bottom after each discussion.


## 1. Project Overview & Goal Reminder

*   **Objective:** Implement and benchmark AlexNet inference (initially Blocks 1 & 2) using a 5-stage approach: V1 (Serial), V2 (MPI CPU), V3 (CUDA GPU), V4 (MPI+CUDA), V5 (CUDA-Aware MPI).
*   **Target Environment:** Fedora 37, GCC 12, CUDA 12.x, Open MPI.
*   **Deliverable:** Working code for (at least) the most advanced version achieved, along with performance analysis comparing the different stages.


## 2. Current Status & Immediate Next Step

*(**Instructions:** Update this section before each meeting)*

*   **Last Completed Stage:** Version 1 (Serial CPU) - Successfully implemented, compiled, and tested the first two blocks (Conv1->ReLU->Pool1, Conv2->ReLU->Pool2->LRN2) using pure C++. Established functional baseline. Code located in `final_project/v1_serial/`.
*   **Current Focus:** **Version 2 (MPI Only - CPU)** - Implementing Approach 2.1 (Broadcast All).
*   **Working Directory:** `final_project/v2_mpi_only/`
*   **Immediate Goal:** Get the basic MPI structure working: initialize MPI, broadcast input data & parameters from Rank 0, have all ranks run the full V1 serial computation locally, extract the final output slice per rank, and gather the slices back to Rank 0 using `MPI_Gatherv`.


## 3. Recent Accomplishments / Milestones Achieved

*(**Instructions:** List key progress made since the last discussion)*

*   Successfully implemented, debugged, and tested the full V1 (Serial CPU) implementation for the first two blocks of AlexNet. Verified correct layer output dimensions and established baseline execution time (~400-500ms on dev machine).
*   Established the 5-version project structure within the `final_project` directory using scaffolding scripts.
*   Configured Makefiles for different compilation needs (`g++` for V1, templates for `mpicxx`/`nvcc`).
*   Set up `ai_context.txt` and refined prompts for efficient AI collaboration.
*   Identified and documented distinct implementation strategies for V2 (Approach 2.1: Broadcast All vs. Approach 2.2: Scatter+Halo).


## 4. Current Challenges / Roadblocks / Issues

*(**Instructions:** List specific technical or conceptual problems you are facing)*

*   **Anticipated for V2.1:**
    *   **`MPI_Bcast` Correctness:** Ensuring `MPI_Bcast` works reliably for `std::vector<float>` data across different MPI implementations/environments. Need to broadcast size first, then data.
    *   **`MPI_Gatherv` Complexity:** Correctly calculating `recvcounts` (size of each rank's final slice) and `displs` (displacement offsets in Rank 0's receive buffer) for potentially non-uniform slice sizes (if `H_final` is not divisible by `num_procs`).
    *   **Indexing for Slice Extraction:** Correctly mapping the 3D indices of the final output map (computed locally) to the 1D indices within the `local_slice` vector for each rank.
    *   **Debugging:** Setting up an efficient workflow for debugging MPI code (e.g., conditional print statements based on rank, using a debugger like `gdb` with multiple processes if possible).
    *   **Relative Path Usage:** Ensuring `../data/` paths work correctly when the MPI executable is launched (depends on `mpirun`'s working directory handling).


## 5. Specific Questions for Professor Sohn

*(**Instructions:** List clear, concise questions. Provide context if needed.)*

1.  **V2 Strategy Validation:** Is the chosen initial V2 strategy (Approach 2.1: Broadcast All, Compute Locally, Gather Final Slice) a sound pedagogical approach for demonstrating MPI understanding for this project stage, or is implementing halo exchanges (Approach 2.2) considered essential early on?
2.  **`MPI_Gatherv` Best Practices:** Are there recommended robust methods or common pitfalls when calculating `recvcounts` and `displs` for `MPI_Gatherv`, especially when dealing with potentially uneven data distribution from slicing feature maps?
3.  **MPI Debugging:** Any specific recommendations for debugging MPI applications on the target cluster (if applicable) or general tips beyond rank-based printing? Are tools like TotalView or DDT available/recommended?
4.  **Submission Scope:** For the final project submission, should we submit code for *all* implemented versions (V1, V2, V3, V4, potentially V5), or primarily the most advanced working version (V4/V5) accompanied by the comparative performance report covering all stages?
5.  **Parameter Handling:** For V2+, is broadcasting all parameters acceptable, or should we explore more advanced distribution if the model grows significantly larger (relevant for future stages)?
6.  **Performance Metrics:** Beyond overall wall-clock time speedup (V2 vs V1, V4 vs V3, etc.), what specific performance metrics (e.g., communication time vs computation time, memory usage per rank) are most important to measure and report for this project?


## 6. Proposed Next Steps / Plan

*(**Instructions:** Outline your plan for the next week or until the next meeting)*

1.  **Implement V2.1 (`v2_mpi_only/`):**
    *   Copy working V1 code (`.cpp`, `.hpp`) to `v2_mpi_only/`. Rename files (`*_serial.cpp` -> `*_mpi.cpp`).
    *   Implement MPI setup/teardown and parameter/data broadcasting in `src/main.cpp`.
    *   Modify `src/alexnet_mpi.cpp` to run the full V1 sequence locally and add logic to extract the final output slice per rank.
    *   Ensure `src/layers_mpi.cpp` contains the correct (unmodified from V1) serial layer logic.
    *   Implement `MPI_Gatherv` logic in `src/main.cpp` on Rank 0.
    *   Update `Makefile` to use `mpicxx` and correct source files.
2.  **Compile & Test V2.1:** Use `make` and `mpirun -np [2, 4, 8] ./template` (with `--oversubscribe` locally).
3.  **Debug V2.1:** Resolve any compilation or runtime errors. Focus on correct data distribution and aggregation.
4.  **(Optional) Verification:** Add code to Rank 0 to compare the gathered MPI result against the V1 serial result for a small test case.
5.  **Begin V3 Planning:** Review V1 code to identify sections suitable for CUDA kernel implementation. Outline V3 (CUDA Only) structure.


## 7. Design Decisions & Considerations

*(**Instructions:** Note significant choices made and alternatives considered)*

*   **V2 Strategy Choice:** Selected Approach 2.1 (Broadcast All) for initial V2 implementation due to its relative simplicity compared to Approach 2.2 (Scatter+Halo). This prioritizes getting a functional MPI version running before tackling complex communication patterns. The trade-off is higher memory usage per rank.
*   **Data Structures:** Continuing to use `std::vector<float>` for simplicity in V1/V2. May reconsider using raw pointers (`float*`) with manual memory management (`new`/`delete[]`) if performance/memory becomes a major issue or for easier integration with CUDA memory allocation in V3/V4.
*   **Intermediate Buffers (V1/V2):** Using a ping-pong buffer approach (`buffer1`, `buffer2`) in the forward pass function to avoid unnecessary data copies between layers.


## 8. Performance & Benchmarking

*(**Instructions:** Add results and analysis here as versions are completed and timed)*

*   **V1 (Serial) Baseline:** ~ [V1 Time, e.g., 477] ms (on [Dev Machine Specs, e.g., WSL2 Ubuntu, CPU Model]) for Blocks 1 & 2.
*   **V2 (MPI):**
    *   Target: Measure wall time using `MPI_Wtime()`.
    *   Metrics: Speedup vs V1 (`T_V1 / T_V2(P)`) for P=2, 4, 8... processes. Scalability analysis.
    *   Results: *(To be added)*
*   **V3 (CUDA):** *(To be added)*
*   **V4 (MPI+CUDA):** *(To be added)*
*   **V5 (CUDA-Aware MPI):** *(To be added)*

---

## Meeting Summaries & Action Items

*(**Instructions:** Manually add notes after each meeting with the professor)*

**YYYY-MM-DD - Discussion with Prof. Sohn**

*   **Topics Discussed:**
    *   [Point 1]
    *   [Point 2]
    *   [Question X answered]
*   **Key Feedback / Decisions:**
    *   [Feedback item 1]
    *   [Decision made on strategy Y]
*   **Action Items:**
    *   [Your action item 1 - Due Date]
    *   [Your action item 2 - Due Date]
    *   [Professor action item (if any)]

---

**YYYY-MM-DD - Discussion with Prof. Sohn**

*   **Topics Discussed:**
    *   ...
*   **Key Feedback / Decisions:**
    *   ...
*   **Action Items:**
    *   ...

---