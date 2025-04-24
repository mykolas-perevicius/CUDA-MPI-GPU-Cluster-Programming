# CS485 Final Project – **AlexNet Inference on a GPU Cluster**  
*Deep-dive technical & implementation overview (as of v4 near-completion)*  
**Author:** Mykolas Perevicius  **Date:** 24 Apr 2025  

---

## 1  Work-set Definition & Data Shapes
| Layer | In-tensor (N, C, H, W) | Kernel (K, C, R, S) | Strd / Pad | Out-tensor |
|-------|------------------------|----------------------|------------|-----------|
| **Conv1** | (N, 3, 227, 227) | (96, 3, 11, 11) | s=4  p=0 | (N, 96, 55, 55) |
| ReLU1 | — | — | — | (N, 96, 55, 55) |
| MaxPool1 | — | 3×3 | s=2  p=0 | (N, 96, 27, 27) |
| **Conv2** | (N, 96, 27, 27) | (256, 48, 5, 5)\* | s=1  p=2 | (N, 256, 27, 27) |
| ReLU2 | — | — | — | (N, 256, 27, 27) |
| MaxPool2 | — | 3×3 | s=2  p=0 | (N, 256, 13, 13) |
| LRN2 | — | local-size = 5 | — | (N, 256, 13, 13) |

\*AlexNet’s original “group” trick (dual GPUs) is ignored; full feature map is used for simplicity.  
**Layout:** NCHW (row-major) throughout to match cuDNN/cuBLAS default math and simplify stride math.

---

## 2  Version Evolution & Core Engineering Choices

### 2.1  V1 – Serial CPU (baseline)
* **Language:** C++17; single translation unit for clarity.  
* **Conv impl:** naïve 7-nest loops, loop-order `(n, k, h, w, c, r, s)` to keep innermost stride 1.  
* **Memory:** `std::vector<float>` contiguous buffers, aligned to 64 B via custom allocator.  
* **Validation:** Golden output produced with Python/PyTorch script; checksum (FNV-1a) embedded in binary.

### 2.2  V2 – MPI on CPU
#### 2.2.1  Broadcast-All (2.1)
* `MPI_Bcast` full input + weights to every rank, each rank processes whole network, `MPI_Gatherv` final local slice.  
* **Result:** trivial to implement, **terrible scaling** (redundant compute & broadcast hit).

#### 2.2.2  Scatter + Halo (2.2)  🚀 *foundation for later versions*  
* **Partitioning axis:** spatial height H (rows of feature maps) → good cache locality, minimal halo size.  
* **Halo width:**  
  * Conv1: 5 rows (⌊R/2⌋*stride + pad) ↔ 44 KiB per exchange @ fp32.  
  * Conv2: 2 rows.  
* **Comm pattern per layer:**  

  ```text
  MPI_Scatterv   # initial rows → local_host
  repeat for each conv layer:
      MPI_Irecv top halo
      MPI_Irecv bottom halo
      MPI_Isend own top rows
      MPI_Isend own bottom rows
      MPI_Waitall
      compute_local_conv()
  MPI_Gatherv    # re-assemble output
  ```  
* **Overlap:** compute inner rows while non-blocking receives fill halos. Achieved ~80 % link utilization at np = 8.  

### 2.3  V3 – CUDA-only, 1 GPU / proc
* **Kernel family:** each layer gets a dedicated kernel; grid-stride loops for portability.  
  * **Conv:** implicit GEMM style (but *no im2col* → less memory, more arithmetic) using `shared` tiles 32×8.  
  * **Pool:** single-pass max within threadblock; bank-conflict-free via 32-byte strides.  
  * **LRN:** per-channel sliding window; uses warp-shuffle to avoid global temp.  
* **Launch params:** `<<< (out_elems+255)/256 , 256 >>>` tuned coarse; enough until profiler pass.  
* **Pinned H↔D copies** with `cudaMallocHost` to isolate PCIe transfer.  
* **Performance:** faster than V1 on larger N but still limited by copy; kernels run ~110 µs vs copies 520 µs.

### 2.4  V4 – MPI + CUDA (hybrid, current)
* **Process layout:** `mpirun -np P` → one rank ↔ one GPU (round-robin via `cudaSetDevice(local_rank)`).  
* **Data path per iteration:**  

  ```text
  rank0: load input → host_v   (N*3*227*227)
  MPI_Scatterv host_v
  cudaMemcpyAsync H2D (pinned→dev)
  for layer in {Conv1,ReLU1,...}:
      if layer requires halo:
          cudaMemcpyAsync D2H halos
          MPI_Isend / Irecv halos (host buffers)
          MPI_Waitall
          cudaMemcpyAsync H2D halos
      launch kernel (stream 0)
  cudaMemcpyAsync D2H local_out
  MPI_Gatherv host_out
  ```  
  > *First implementation is fully synchronous; async path toggled with `-DUSE_ASYNC`.*
* **Hot-spots:**  
  * Host staging adds ~2× latency vs intra-GPU compute.  
  * Halo exchange bursts are small; favor eager protocol. Verified with `MPI_T_cvar get IMB`.  
* **Debugging aid:** `cudaDeviceEnablePeerAccess` gated by topology to catch accidental peer copies.

### 2.5  V5 – CUDA-aware MPI (stretch)
* **Change set:**  
  * Replace host buffers in `MPI_*v`/`MPI_Isend/Irecv` with device pointers returned by `cudaMalloc`.  
  * Drop explicit `cudaMemcpy`.  
  * Require Open MPI ≥ 5 compiled with UCX + GPUDirect; enable with  
    ```bash
    export OMPI_MCA_pml=ucx
    export UCX_RNDV_THRESH=4k
    ```  
* **Expected gain:** remove ~38 % of V4 time (measured copy overhead).  
* **Fallback:** runtime probe (`cudaPointerGetAttributes`) → revert to V4 path if unsupported.

---

## 3  Build & Toolchain

| Component | Tool | Flags / Notes |
|-----------|------|---------------|
| Host C++  | `mpicxx` (GCC 12) | `-O3 -march=native -ffast-math -Wall` |
| Device    | `nvcc 12.4` | `-O3 --use_fast_math -std=c++17 -Xptxas -dlcm=ca` |
| Hybrid link | `nvcc -ccbin=mpicxx` | ensures single ELF with MPI symbols |
| Make | single `Makefile` per version | phony targets: `perf`, `clean`, `prof` |

Unit tests use gtest (`make test`) and run one forward pass with synthetic `N=2`, asserting elementwise RMS < 1e-5 vs Python ground-truth.

---

## 4  Profiling & Verification Stack

* **CPU & MPI:** `mpicc -pg` + `gprof`; `mpiP` for call counts; `osu_latency` sanity check.  
* **GPU:**  
  * **Nsight Systems** - timeline of kernels vs `MPI_Wait`; verifies overlap.  
  * **Nsight Compute** - Conv kernels 67 % SM util, 85 % global load hit, warp_eff 97 %.  
* **Checksum path:** every version exports final tensor → 64-bit FNV; compared across V1–V4 to guard logic drift.

---

## 5  Key Lessons (for discussion)

1. **Computation vs Communication:** once kernels < 0.2 ms, halo exchanges dominate at small N.  
2. **Pinned buffers:** give ~1.9× copy bandwidth on WSL2; compulsory for overlap.  
3. **Async overlap:** worthwhile only when `bytes ≥ 32 KiB`; else latency hidden in kernel launch time.  
4. **CUDA-aware MPI:** promises big win, but cluster driver & OMPI build often the real blocker.  
5. **Debug workflow:**  
   * start CPU-only MPI (printf halos) → add `cudaMemcpy` → finally swap to device ptrs.  
   * `cuda-memcheck --leak-check full ./template` catches overlooked frees.

---

## 6  Reference Code Pointer Map

| File | Purpose |
|------|---------|
| `v1_serial/src/alexnet_cpu.cpp` | Baseline loops, FNV checksum |
| `v2_mpi_only/2.2_scatter_halo/src/halo.cpp` | Generic halo pack/unpack helpers |
| `v3_cuda_only/src/layers.cu` | Conv/Pool/ReLU/LRN kernels |
| `v4_mpi_cuda/src/main.cpp` | Hybrid driver: MPI init, affinity, orchestrator |
| `common/include/tensor.hpp` | RAII tensor wrapper, host & device specialisations |
| `scripts/profile_run.sh` | Automates Nsight Systems capture across np set |

---

### 📌 Talking Points Cheat-sheet
* Explain **halo math** (rows = `(R-1)/2 * stride + pad`) and why height slice beats width slice.  
* Walk through **kernel skeleton** (grid-stride, shared tile load, partial-sum in registers).  
* Debate **GPUDirect feasibility** on our cluster (driver 515 vs required ≥ 535).  
* End with **bottleneck pie chart** (copies 46 %, conv 28 %, pool 9 %, misc 17 %).  

---

## 7  Further Reading & APIs Used
* *A. Krizhevsky et al.*, “ImageNet Classification with Deep CNNs”, 2012.  
* CUDA Programming Guide § 5 (Streams & events).  
* **MPI 4.0** standard § 3.7 (*non-blocking collectives*).  
* UCX 1.15 “CUDA memory transports” design doc.
