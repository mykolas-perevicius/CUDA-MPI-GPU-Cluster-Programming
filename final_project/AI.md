# AI Context Document: CS485 Final Project - AlexNet Inference (MPI+CUDA)

**Version:** 1.0 (Updated based on V1-V4 Code Review & Script Output)
**Purpose:** Provide comprehensive context for AI assistants regarding the structure, status, implementation details, challenges, and goals of the CS485 final project. This aims to minimize the need for re-explaining core concepts or re-pasting large code sections in future interactions.

## 1. Project Overview & Scope

*   **Objective:** Implement and benchmark inference for the **first two blocks of the AlexNet CNN** (Conv1->ReLU->Pool1 -> Conv2->ReLU->Pool2->LRN2) across a series of parallelization stages (V1-V5).
*   **Goal:** Learn and compare different parallel programming paradigms (Serial, MPI, CUDA, Hybrid MPI+CUDA) and analyze their performance characteristics (speedup, scalability, bottlenecks) on this representative workload. Full AlexNet implementation is an optional extension.
*   **Core Task:** Produce working code for each stage (V1-V5), perform detailed performance analysis, and present findings.
*   **Target Environment:** Fedora 37, GCC 12, CUDA 12.x, Open MPI (ideally CUDA-aware for V5). Development occurs in WSL2 (Ubuntu) but final validation must be on the target platform.

## 2. Overall Project Structure

*   **Staged Development (V1-V5):** Incremental approach adding complexity:
    *   V1: Serial CPU (Baseline)
    *   V2: MPI Only (CPU cores)
    *   V3: CUDA Only (Single GPU)
    *   V4: MPI + CUDA Hybrid (Host Staging)
    *   V5: MPI + CUDA Hybrid (CUDA-Aware MPI optimization)
*   **Directory Structure:** Organized by version (`final_project/v1_serial/`, `v2_mpi_only/`, etc.), with sub-directories for V2 approaches. Shared data/docs accessed via relative paths. Key files: `README.md`, `RESEARCH.md`, `discussion.md`, `ai_context.txt` (this file), Makefiles, source (`src/`), includes (`include/`).
    ```
    final_project/
    ├── v1_serial/       (Complete)
    ├── v2_mpi_only/
    │   ├── 2.1_broadcast_all/ (Complete)
    │   └── 2.2_scatter_halo/  (Complete)
    ├── v3_cuda_only/    (Complete)
    ├── v4_mpi_cuda/     (Implemented - Debugging)
    ├── v5_cuda_aware_mpi/ (Pending - Baseline Code)
    ├── data/
    ├── docs/
    ├── logs/
    ├── scripts/         (Contains run_final_project.sh)
    └── ... (Documentation files)
    ```
*   **Build System:** Standard Makefiles per version.
    *   V1: `g++`
    *   V2: `mpicxx`
    *   V3: `nvcc`
    *   V4/V5: `nvcc -ccbin=mpicxx` (compiles both .cu and .cpp, links MPI/CUDA). V4 Makefile includes `bear` integration for `compile_commands.json` generation and `clang-tidy` for linting.
    *   Key Flags: `-std=c++11/17`, `-O3`, `-Wall`, MPI include/link flags (via `mpicxx --showme`), CUDA arch flags (`-gencode arch=compute_75,code=sm_75`).

## 3. Data Structures & Parameters

*   **`LayerParams` Struct:** Holds parameters for a logical block (e.g., Conv+Pool+LRN). Defined in `include/alexnet.hpp` (slight variations between versions). Host vectors used for storage.
    ```c++
    // Representative Structure (V1/V3 style)
    struct LayerParams {
        std::vector<float> weights; // Host vectors for weights
        std::vector<float> biases;  // Host vectors for biases
        int K, F, S, P;       // Conv params (Kernels, FilterSize, Stride, Padding)
        int F_pool, S_pool;   // Pooling params
        int N_lrn;            // LRN params
        float alpha, beta, k_lrn;
    };
    ```
*   **Data Representation:** Primarily uses `std::vector<float>` on the host and raw `float*` pointers (allocated via `cudaMalloc`) on the CUDA device.
*   **Key Dimensions/Params (Consistent):**
    *   Input: H=227, W=227, C=3
    *   Conv1: K=96, F=11, S=4, P=0
    *   Pool1: F=3, S=2
    *   Conv2: K=256, F=5, S=1, P=2 (Input Channels = Conv1 K = 96)
    *   Pool2: F=3, S=2
    *   LRN2: N=5, alpha=1e-4, beta=0.75, k=2.0
    *   Final Output Shape (Expected): H=13, W=13, C=256 (Conv2 K)

## 4. Implementation Strategies & Status per Version

### 4.1 Version 1: Serial CPU (Completed)
*   **Strategy:** Straightforward C++ implementation using nested loops for convolution, pooling, etc. Operates on `std::vector<float>`. Serves as functional baseline and correctness reference.
*   **Key Files:** `v1_serial/src/{main.cpp, alexnet_serial.cpp, layers_serial.cpp}`.

### 4.2 Version 2: MPI Only (CPU) (Completed)
*   **Goal:** Parallelize V1 using MPI across CPU cores. Two approaches implemented.
*   **V2.1 (Broadcast All):**
    *   Strategy: Rank 0 broadcasts full input & parameters. All ranks compute the *entire* V1 sequence locally. Rank 0 gathers the final slice needed from each rank.
    *   Outcome: Simple but scales poorly (communication bottleneck, redundant computation).
*   **V2.2 (Scatter + Halo):**
    *   Strategy: Rank 0 scatters input rows (`MPI_Scatterv`). Ranks exchange halo regions (boundary rows needed for convolution) using non-blocking `MPI_Isend`/`Irecv`/`Wait`. Parameters broadcast. Ranks compute V1 sequence on their local data + halos. Asymmetric trimming applied after pooling layers to remove halo influence. Final results gathered (`MPI_Gatherv`).
        ```c++
        // Snippet: V2.2 Halo Exchange Concept (main.cpp)
        // ... scatter input to localIn ...
        int pad1 = conv1.F / 2;
        int slice1 = pad1 * W * C;
        std::vector<float> topHalo(slice1), botHalo(slice1);
        MPI_Request reqs; int q=0;
        if (rank > 0) {
            MPI_Irecv(topHalo.data(), slice1, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &reqs[q++]);
            MPI_Isend(localIn.data(), slice1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &reqs[q++]);
        } // else fill topHalo with 0
        if (rank < size - 1) {
            MPI_Irecv(botHalo.data(), slice1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &reqs[q++]);
            MPI_Isend(localIn.data() + (localH - pad1) * W * C, slice1, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &reqs[q++]);
        } // else fill botHalo with 0
        MPI_Waitall(q, reqs, MPI_STATUSES_IGNORE);
        // ... construct padded buffer using localIn, topHalo, botHalo ...
        // ... compute layers ...
        // ... trim result ...
        // ... gather ...
        ```
    *   Outcome: More complex but scales well, demonstrating effective MPI communication pattern. Foundation for V4.
*   **Key Files:** `v2_mpi_only/2.2_scatter_halo/src/{main.cpp, alexnet_mpi.cpp, layers_mpi.cpp}` (V2.2 is primary).

### 4.3 Version 3: CUDA Only (Single GPU) (Completed)
*   **Strategy:** Port V1 layer logic to CUDA kernels. Host code manages data lifecycle: allocate device memory (`cudaMalloc`), copy input/weights H->D (`cudaMemcpyHostToDevice`), launch sequence of kernels (Conv1, ReLU1, Pool1, ... LRN2), copy final result D->H (`cudaMemcpyDeviceToHost`), free device memory (`cudaFree`). Basic 1D grid-stride kernels.
    ```c++
    // Snippet: V3 Execution Flow Concept (alexnet_cuda.cu)
    // ... calculate dims ...
    // ... cudaMalloc all device buffers (d_input, d_c1out, d_p1out, etc., d_weights, d_biases) ...
    // ... cudaMemcpy H->D input_host -> d_input, params -> d_weights/d_biases ...

    // Layer Sequence Execution:
    cudaConvLayer(d_c1out, d_input, d_w1, d_b1, ...);
    cudaReluLayer(d_c1out, c1_sz);
    cudaMaxPoolLayer(d_p1out, d_c1out, ...);
    cudaConvLayer(d_c2out, d_p1out, d_w2, d_b2, ...);
    cudaReluLayer(d_c2out, c2_sz);
    cudaMaxPoolLayer(d_p2out, d_c2out, ...);
    cudaLRNLayer(d_l2out, d_p2out, ...);

    // ... cudaMemcpy D->H d_l2out -> output_host ...
    // ... cudaFree all device buffers ...
    ```
*   **Outcome:** Functionally complete. Performance is currently *worse* than V1 serial, indicating high H<->D transfer overhead or inefficient kernels. Needs profiling (`Nsight Systems`/`Compute`). Numerical output differs from V1/V2.
*   **Key Files:** `v3_cuda_only/src/{main_cuda.cpp, alexnet_cuda.cu, layers_cuda.cu}`.

### 4.4 Version 4: MPI + CUDA (Hybrid) (Implemented - Needs Debugging)
*   **Strategy:** Combine V2.2 MPI data distribution (Scatter+Halo) with V3 CUDA computation. **Crucially, uses explicit Host Staging.**
*   **Execution Flow (`main_mpi_cuda.cpp` calling `alexnetTileForwardCUDA`):**
    1.  MPI Setup: Parameters broadcast, input rows scattered to host buffers (`myIn`).
    2.  **Host Halo Exchange:** Conv1 halo regions exchanged via `MPI_Isend/Irecv/Wait` into host buffers (`recvT`, `recvB`). `myIn` is then conceptually padded on host (though implementation inserts into `std::vector`).
    3.  **Host -> Device Transfer:** The *entire padded* host buffer `myIn` is copied to device memory `d_in` (`cudaMemcpyHostToDevice`).
    4.  **GPU Tile Computation:** Call `alexnetTileForwardCUDA(d_in, ..., d_out)`. This helper function (in `alexnet_mpi_cuda.cu`) internally allocates intermediate device buffers (for conv1, pool1, etc.), allocates+copies weights/biases H->D *within the function*, launches the V3 kernel sequence (Conv1..LRN2), and frees internal buffers. The final result is placed in `d_out`.
        ```c++
        // Snippet: V4 GPU Tile Computation Call (main_mpi_cuda.cpp)
        // ... Host halo exchange into myIn ...
        float *d_in=nullptr; CUDA_CHECK(cudaMalloc(&d_in, myIn.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, myIn.data(), myIn.size()*sizeof(float), cudaMemcpyHostToDevice));
        // ... Calculate final output tile size (Hp2, Wp2, p2.K) ...
        std::vector<float> tileOut((size_t)Hp2 * Wp2 * p2.K); // Host buffer for D->H result
        float* d_out; CUDA_CHECK(cudaMalloc(&d_out, tileOut.size()*sizeof(float))); // Device buffer for final tile output

        // Runs ENTIRE sequence Conv1..LRN2 on the GPU for the padded tile d_in
        alexnetTileForwardCUDA(d_in, p1, p2, (int)myIn.size()/rowSz, W, C, d_out);

        CUDA_CHECK(cudaMemcpy(tileOut.data(), d_out, tileOut.size()*sizeof(float), cudaMemcpyDeviceToHost));
        // ... Host trimming of tileOut into local ...
        // ... Gather local ...
        cudaFree(d_in); cudaFree(d_out);
        ```
    5.  **Device -> Host Transfer:** Final result `d_out` is copied back to host buffer `tileOut` (`cudaMemcpyDeviceToHost`).
    6.  **Host Trimming:** Halo-related rows are removed from `tileOut` on the host to create `local` result (current logic seems simplified).
    7.  MPI Gather: `local` results are gathered (`MPI_Gatherv`).
    8.  GPU Affinity: `cudaSetDevice(rank % nDev)` used.
*   **Current Status:** Implemented but debugging needed.
*   **Key Issues:**
    *   **Output Format:** `cout` statements in `main_mpi_cuda.cpp` don't match `run_final_project.sh` expectations, causing parsing warnings (`⚠`). Needs string changes.
    *   **NP=4 Crash:** Fails with MPI Exit Code 134 when run with 4 processes. Log file `logs/final_project_v4_np4.log` needs inspection. Likely resource issue, memory error, or MPI communication bug triggered at np=4.
    *   **Correctness/Trim:** Numerical output needs verification. Host trimming logic correctness needs review.
*   **Key Files:** `v4_mpi_cuda/src/{main_mpi_cuda.cpp, alexnet_mpi_cuda.cu, layers_mpi_cuda.cu}`. Note the existence of an unused, more complex function `alexnetForwardPassMPI_CUDA` in `alexnet_mpi_cuda.cu`.

### 4.5 Version 5: CUDA-Aware MPI (Pending)
*   **Goal:** Optimize V4 by eliminating explicit host staging *for MPI communication calls* (`Scatterv`, `Isend`, `Irecv`, `Gatherv`).
*   **Strategy:** Modify V4 MPI calls to pass **device pointers** directly. Requires CUDA-aware Open MPI build and potentially GPUDirect RDMA support on the cluster hardware/network for maximum benefit. If RDMA isn't available, the library might internally stage through host memory, reducing benefits.
*   **Status:** Baseline code exists, implementation pending V4 stabilization.

## 5. Performance Summary & Observations (WSL2 Dev Env - Script Output)

| Version                | Procs | Shape     | Time       | Status | Key Observation                             |
| :--------------------- | :---- | :-------- | :--------- | :----- | :------------------------------------------ |
| V1 Serial              | 1     | 13x13x256 | ~667 ms    | ✔      | Baseline                                    |
| V2 2.1 Broadcast       | 4     | 13x13x256 | ~881 ms    | ✔      | Poor scaling (Broadcast/Compute overhead) |
| V2 2.2 Scatter+Halo  | 4     | 13x13x256 | ~281 ms    | ✔      | Good scaling (Efficient MPI comm)         |
| V3 CUDA                | 1     | 13x13x256 | ~2349 ms   | ✔      | Slow (H<->D / Kernel Bottleneck?)         |
| V4 MPI+CUDA          | 1     | –         | –          | ⚠      | Runs, Output format mismatch              |
| V4 MPI+CUDA          | 2     | –         | –          | ⚠      | Runs, Output format mismatch              |
| V4 MPI+CUDA          | 4     | –         | –          | ⚠      | CRASHES (Exit 134)                          |

*   V2.2 shows significant speedup over V1 and V2.1.
*   V3 is unexpectedly slow; needs profiling.
*   V4 is functionally incomplete due to output/crash issues. Performance cannot be assessed yet.
*   Significant numerical differences observed in sample outputs between V1, V2.1, V2.2(np>1), and V3 require investigation.

## 6. Immediate Debugging Tasks (V4)

1.  **Fix Output Formatting:** In `v4_mpi_cuda/src/main_mpi_cuda.cpp`, modify `std::cout` lines in the `rank == 0` block to match the expected format:
    *   `shape HxWxK` -> `Final Output Shape: HxWxK`
    *   `sample v1 v2 ...` -> `Final Output (first 10 values): v1 v2 ...`
    *   Add time output: `std::cout << "AlexNet MPI+CUDA Forward Pass completed in " << duration_ms << " ms" << std::endl;` (and update `run_final_project.sh` to parse it).
2.  **Diagnose NP=4 Crash:**
    *   Analyze `logs/final_project_v4_np4.log` for MPI/CUDA error messages.
    *   Re-run manually: `mpirun -np 4 ./template`.
    *   If needed, use debuggers (`gdb --args mpirun ...`, `cuda-gdb`) or memory checkers (`cuda-memcheck mpirun ...`). Check for out-of-bounds access, incorrect MPI counts/displacements, resource exhaustion.
3.  **Verify Correctness:**
    *   Once V4 output is parseable, compare numerical values against V1/V3 reference outputs (allow for floating-point tolerance).
    *   Review the host trimming logic in `main_mpi_cuda.cpp` (using `start`/`stop` variables) - ensure it correctly removes only the rows corresponding to the effective halo contribution after the full layer sequence. Consider edge cases (small inputs, different NP values).

## 7. Future Directions

*   Complete V4 debugging and validation.
*   Profile V3 and V4 using `Nsight Systems` / `Nsight Compute` to identify and address performance bottlenecks.
*   Implement and benchmark V5 (CUDA-Aware MPI) if cluster environment permits.
*   Investigate numerical differences between versions.
*   Explore implementing the alternative, potentially more complex logic in the unused `alexnetForwardPassMPI_CUDA` function in V4.
*   Consider performance optimizations (async overlap, kernel tuning) based on profiling.

---