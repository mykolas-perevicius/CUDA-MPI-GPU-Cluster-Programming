# TODO for Project Version: v2_mpi_only

This directory contains code copied from the V4 MPI+CUDA version and has undergone **BASIC AUTOMATED MODIFICATION**.
Significant **MANUAL WORK** is required to make it functional for version **v2_mpi_only**.

**Automated Changes Attempted:**
*   Lines containing common CUDA functions/includes/keywords were commented out (prefixed with `// CUDA? `). **REVIEW THESE CAREFULLY.**
*   `.cu` files in `src/` containing `__global__` were renamed to `.cpp.stub`. You need to implement serial logic in them and likely rename to `.cpp`.
*   Placeholder comments (`// TODO: Implement serial logic...`) were added near likely CUDA kernel calls.
*   Attempted to change data/doc paths from `../final_project/data/` to `../data/`. **VERIFY PATHS.**

## Shared Resources Access:

*   Shared data/docs reside in parent `final_project` directory (e.g., `../data/`, `../docs/`).
*   **Action Required:** Ensure file paths within the source code (`final_project/v2_mpi_only/src/*`) correctly reference shared resources using relative paths like `../data/`.

## Specific Manual Steps for Version 'v2_mpi_only':

1.  **Review Comments:** Carefully review all lines commented out with `// CUDA? `. Delete them if the removal is correct, or uncomment/fix if needed.
2.  **Implement Serial Logic per Rank:** Find all `.cpp.stub` files and `// TODO: Implement serial logic...` comments. Replace the commented-out CUDA kernel functionality with **correct serial C++ CPU code that runs within each MPI rank**.
3.  **Rename Files:** Once serial logic is implemented, rename `.cpp.stub` files to `.cpp`.
4.  **Clean Includes/Headers:** Remove unnecessary CUDA includes. Update header prototypes in `include/`. Keep MPI includes.
5.  **Update Makefile:** Edit `Makefile`. **CRITICALLY, update the `SRCS = ...` list** to include only the final, correctly named `.cpp` files. Ensure compiler is `mpicxx` and flags/libs are correct.
6.  **Verify Data Paths:** Double-check all paths accessing `../data/`.
