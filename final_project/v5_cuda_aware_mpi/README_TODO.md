# TODO for Project Version: v5_cuda_aware_mpi

This directory contains the codebase restored from the backup 'original_backup_20250422_012349'.
It should represent the working MPI+CUDA version.

## Shared Resources Access:

*   Shared data/docs reside in parent `final_project` directory (e.g., `../data/`, `../docs/`).
*   **Action Required:** Verify file paths within the source code (`final_project/v5_cuda_aware_mpi/src/*.cpp/cu`) correctly reference shared resources using relative paths like `../data/`.

## Specific Manual Steps for Version 'v5_cuda_aware_mpi':

1.  **Verify Code:** Base MPI+CUDA code should be correct.
2.  **CUDA-aware MPI Calls:** **(Optional/Advanced)** Modify MPI communication calls in `src/` to use GPU device pointers directly. Requires CUDA-aware MPI library setup.
3.  **Verify Makefile:** Check compiler (`nvcc -ccbin=mpicxx`), flags, linked libraries. Ensure MPI environment supports CUDA-awareness.
4.  **Verify Data Paths:** Ensure code uses relative paths like `../data/...`.
