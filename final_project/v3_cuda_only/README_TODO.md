# TODO for Project Version: v3_cuda_only

This directory contains code copied from the V4 MPI+CUDA version and has undergone **BASIC AUTOMATED MODIFICATION**.
Significant **MANUAL WORK** is required to make it functional for version **v3_cuda_only**.

**Automated Changes Attempted:**
*   Lines containing common MPI functions/includes were commented out (prefixed with `// MPI? `). **REVIEW THESE CAREFULLY.**
*   Attempted to change data/doc paths from `../final_project/data/` to `../data/`. **VERIFY PATHS.**

## Shared Resources Access:

*   Shared data/docs reside in parent `final_project` directory (e.g., `../data/`, `../docs/`).
*   **Action Required:** Ensure file paths within the source code (`final_project/v3_cuda_only/src/*`) correctly reference shared resources using relative paths like `../data/`.

## Specific Manual Steps for Version 'v3_cuda_only':

1.  **Review Comments:** Carefully review all lines commented out with `// MPI? `. Delete them if the removal is correct, or uncomment/fix if needed. The program must run as a single process.
2.  **Adjust Logic:** Ensure CUDA code processes the *entire* dataset (no MPI data slicing).
3.  **Clean Includes/Headers:** Remove unnecessary MPI includes. Update header prototypes if needed. Keep CUDA includes.
4.  **Update Makefile:** Edit `Makefile`. **CRITICALLY, update the `SRCS_CU` / `SRCS_CPP` lists.** Ensure compiler is `nvcc` and flags (`-gencode`) / libs (`-lcudart`) are correct.
5.  **Verify Data Paths:** Double-check all paths accessing `../data/`.
