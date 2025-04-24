# AI Context for CS485 Final Project: AlexNet Inference (MPI+CUDA) - V1, V2, V3 Complete

## 1. Project Scope & Status
- **Target:** Staged implementation (V1-V5) of AlexNet **Blocks 1 & 2** (Conv1->ReLU->Pool1->Conv2->ReLU->Pool2->LRN2). Full network is an extension.
- **Completed:** V1 (Serial), V2 (MPI Only - both 2.1_Broadcast & 2.2_Scatter+Halo approaches), V3 (CUDA Only).
- **Next Step:** Implement **V4 (MPI+CUDA Hybrid)**, integrating V2.2 MPI logic with V3 CUDA kernels.

## 2. Environment & Build
- **Target:** Fedora 37, GCC 12, CUDA 12.x, Open MPI (CUDA-aware needed for V5).
- **Dev:** WSL2 (Ubuntu), GCC 12, Open MPI, CUDA 12.8.
- **Build Tools:** Makefiles specific to each version/sub-version (`g++`, `mpicxx`, `nvcc`, `nvcc -ccbin=mpicxx`).

## 3. Key Code Structures & Parameters

**`LayerParams` Struct (`include/alexnet.hpp` - Used across V1-V3):**
```c++
struct LayerParams {
    std::vector<float> weights; // Host vectors
    std::vector<float> biases;  // Host vectors
    int K, F, S, P;       // Conv params
    int F_pool, S_pool;   // Pooling params
    int N_lrn;            // LRN params
    float alpha, beta, k_lrn;
    // Note: V3+ use float* device pointers for computation.
};
```

**V1 Serial (`v1_serial/`):** Pure C++, `std::vector`, direct loops. Baseline functional code.

**V2 MPI Approaches (`v2_mpi_only/`):**
*   **2.1_broadcast_all:** `MPI_Bcast` full input/params. All ranks compute full V1 sequence locally. Gather final slice via `MPI_Gatherv`. **Simple but poor scaling.**
*   **2.2_scatter_halo:** `MPI_Scatterv` input rows. Halo exchange via `MPI_Isend`/`Irecv`/`Wait`. `MPI_Bcast` params. Ranks compute on local data+halo using V1 serial logic. Gather final slice via `MPI_Gatherv`. **More complex, better scaling.**

**V3 CUDA (`v3_cuda_only/`):**
*   Kernels (`__global__`) for layers in `layers_cuda.cu` (simple 1D grid-stride).
*   Host manages GPU memory (`cudaMalloc`/`cudaMemcpy`/`cudaFree`) and kernel launches in `alexnet_cuda.cu`. `CUDA_CHECK` macro used.

**Example Parameters Used (Consistent across V1-V3):** Input H=227, W=227, C=3; Conv1 K=96, F=11, S=4, P=0; Pool1 F=3, S=2; Conv2 K=256, F=5, S=1, P=2; Pool2 F=3, S=2; LRN2 N=5, alpha=1e-4, beta=0.75, k=2.0.

## 4. Performance Summary Table (Blocks 1&2, WSL2 Dev Machine)
```
╔════════════════════════╤═══════╤═════════════╤════════════╤═════╗
║  Version                ║ Procs ║ Shape       ║ Time       ║ St  ║
╟════════════════════════┼═══════┼═════════════┼════════════┼═════╢
║ V1 Serial              ║     1 ║ 13x13x256   ║   ~617 ms  ║  ✔  ║
║ V2 2.1-broadcast-all   ║     1 ║ 13x13x256   ║  ~660 ms   ║  ✔  ║
║ V2 2.1-broadcast-all   ║     2 ║ 13x13x256   ║  ~704 ms   ║  ✔  ║
║ V2 2.1-broadcast-all   ║     4 ║ 13x13x256   ║  ~802 ms   ║  ✔  ║
║ V2 2.2-scatter-halo    ║     1 ║ 13x13x256   ║  ~491 ms   ║  ✔  ║
║ V2 2.2-scatter-halo    ║     2 ║ 13x13x256   ║  ~334 ms   ║  ✔  ║
║ V2 2.2-scatter-halo    ║     4 ║ 13x13x256   ║  ~177 ms   ║  ✔  ║
║ V3 CUDA                ║     1 ║ 13x13x256   ║  ~750 ms   ║  ✔  ║
╚════════════════════════╧═══════╧═════════════╧════════════╧═════╝
```
*(Note: V3 needs profiling; H<->D or kernel optimization likely needed.)*

## 5. Immediate Task: Implement V4 (MPI + CUDA Hybrid)
- **Location:** `final_project/v4_mpi_cuda/`
- **Goal:** Integrate **V2.2 MPI logic (Scatterv, Halo Exchange, Gatherv)** with **V3 CUDA kernels**.
- **Strategy:**
    1.  Base code: Use restored V4/V5 code, heavily referencing V2.2 and V3 implementations.
    2.  MPI handles overall structure, data distribution (Scatterv to *host* buffers), halo communication (*host* buffers), result aggregation (Gatherv from *host* buffers).
    3.  CUDA handles layer computation on GPU.
    4.  **Host Staging Required:** Explicit `cudaMemcpy H2D` needed after Scatterv/halo Recv. Explicit `cudaMemcpy D2H` needed before halo Send / final Gatherv.
    5.  Launch CUDA kernels from V3 (`layers_cuda.cu`) on device data. Use device pointers (`float*`) for kernel args.
    6.  Manage GPU affinity (`cudaSetDevice`).
    7.  Synchronize MPI/CUDA (e.g., `MPI_Wait` for halo, `cudaDeviceSynchronize` or Events before D2H).
    8.  Build with `nvcc -ccbin=mpicxx`.
- **Key Files:** `Makefile`, `include/alexnet.hpp`, `include/layers.hpp`, `src/main.cpp` (MPI focus), `src/alexnet_mpi_cuda.cu` (Hybrid orchestration, H<->D copies), `src/layers_cuda.cu` (Kernels - likely copied from V3).

## 6. Future Task: V5 (CUDA-Aware MPI)
- **Location:** `final_project/v5_cuda_aware_mpi/`
- **Goal:** Optimize V4 by removing host staging.
- **Strategy:** Modify V4 MPI calls to use *device* pointers directly. Requires CUDA-aware MPI library. Compare performance vs V4.
