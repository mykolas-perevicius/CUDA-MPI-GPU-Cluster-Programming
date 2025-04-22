# TODO for Project Version: v4_mpi_cuda

This directory contains the codebase restored from the backup 'original_backup_20250422_012349'.
It should represent the working MPI+CUDA version.

## Shared Resources Access:

*   Shared data/docs reside in parent `final_project` directory (e.g., `../data/`, `../docs/`).
*   **Action Required:** Verify file paths within the source code (`final_project/v4_mpi_cuda/src/*.cpp/cu`) correctly reference shared resources using relative paths like `../data/`.

## Specific Manual Steps for Version 'v4_mpi_cuda':

1.  **Verify Code:** Ensure MPI and CUDA calls are present and coordinated correctly.
2.  **Data Transfers:** Check host <-> device copies (`cudaMemcpy`) are correct for MPI.
3.  **Verify Makefile:** Check compiler (`nvcc -ccbin=mpicxx`), flags, linked libraries (`cudart`), and source file list.
4.  **Verify Data Paths:** Ensure code uses relative paths like `../data/...`.
