# CS485: GPU Cluster Programming (MPI+CUDA) – Project Workspace

This repository is the central hub for our CS485 GPU Cluster Programming course at NJIT. It includes homework assignments, final project code (an evolving AlexNet inference implementation), automation scripts, and additional resources. Everything is designed to streamline the development, testing, and submission workflows for MPI and CUDA programs running on Fedora 37 with GCC 12 and CUDA Toolkit 12.x.

## Table of Contents
1. [Course Overview](#course-overview)
2. [Project Objectives](#project-objectives)
3. [Key Technologies & Environment](#key-technologies--environment)
4. [Repository Structure](#repository-structure)
5. [Current Implementation Status](#current-implementation-status)
6. [Development Workflow](#development-workflow)
7. [Automation Scripts](#automation-scripts)
8. [Build & Test Instructions](#build--test-instructions)
9. [Submission Guidelines](#submission-guidelines)
10. [Troubleshooting](#troubleshooting)
11. [Future Directions](#future-directions)
12. [References & Resources](#references--resources)

## 1. Course Overview
**Course:** CS485 – Selected Topics: GPU Cluster Programming (MPI+CUDA)  
**Instructor:** Andrew Sohn, NJIT  
**Topics:** 
- High-performance computing (HPC) fundamentals
- Parallel architectures (MIMD, SIMD, SPMD)
- MPI for distributed memory systems (Point-to-Point, Collective, One-Sided)
- CUDA for GPU acceleration (architecture, memory management, kernel optimization)
- Data-parallel approaches on a GPU cluster (MPI + CUDA)
- Real-world HPC workflows, debugging, and performance tuning

**Required Grading Environment:** 
- **OS:** Fedora 37  
- **Compilers:** GCC 12  
- **GPU Toolkit:** CUDA 12.x  

All final submissions must compile and run under this environment, using an automated grading script.

**Relevant Links:**
- [Course Webpage](http://web.njit.edu/~sohna/cs485)
- [MPI v3.1 Standard](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)
- [Programming Massively Parallel Processors (4th Ed.)](https://www.elsevier.com/books/programming-massively-parallel-processors-a-hands-on-approach/kirk/978-0-323-91231-0)

## 2. Project Objectives
1. **Complete Homework Assignments:** Tackle progressive homework tasks (MPI basics, CUDA kernels, advanced MPI+CUDA), ensuring correctness and performance.
2. **Develop a Final Project:** An AlexNet-based inference engine that demonstrates multi-GPU parallelism using MPI+CUDA, achieving tangible speedup over single-GPU or single-node baselines.
3. **Automate Everything:** Provide scripts for scaffolding, building, testing, and packaging to ensure reproducible results and consistency with the grading environment.
4. **Maintain Clear Documentation:** Keep each component thoroughly documented (comments, readmes, script usage guides).
5. **Use AI Assistance for Learning:** Leverage AI chat for code summaries, potential exam questions, debugging, and deeper HPC insights.

## 3. Key Technologies & Environment
- **MPI (Open MPI)**: For distributing computation across multiple Linux nodes or multiple ranks on one node. Key functions include `MPI_Init`, `MPI_Bcast`, `MPI_Allreduce`, etc.
- **CUDA (NVIDIA)**: For GPU kernel development, memory transfers, and HPC acceleration. Key toolkit components include `nvcc`, `cudaMemcpy`, `cudaEvent`, and advanced libraries (cuBLAS/cuDNN).
- **Fedora 37** with **GCC 12** and **CUDA 12.x**: Mandatory environment for final grading. We simulate or partially test this environment locally (WSL2 with Ubuntu + custom GCC 12 + CUDA 12.x).
- **Bash & Make**: For automation and build workflows. Some homeworks also generate CMake files for local convenience.
- **AI Tools**: Used for generating code explanations, debugging hints, and thorough documentation.

## 4. Repository Structure
```
.
├── README.md                # This readme
├── homeworks/              # Homework assignments
│   └── hwX/
│       ├── src/           # Source code (template.c/.cu)
│       ├── Makefile       # Generated makefile for submission
│       ├── CMakeLists.txt # For local dev (HW4+)
│       ├── build/         # Local build folder (ignored in submission)
│       └── summary.md     # Summaries, notes, potential exam Qs
├── final_project/         # Final project: AlexNet MPI+CUDA
│   ├── include/          # Headers (layers.hpp, alexnet.hpp, etc.)
│   ├── src/             # Source (main.cpp, alexnet_hybrid.cu, layers.cu)
│   ├── Makefile         # Build instructions (produces 'template', etc.)
│   └── scripts/         # Additional scripts (test_final.sh, etc.)
├── scripts/              # Global automation scripts
│   ├── scaffold_hw.sh   # Creates new homework structure
│   ├── test_hw.sh      # Builds/tests HW with local run
│   ├── package_hw.sh   # Packages HW for submission
│   ├── run_hw.sh       # Combined test + package
│   ├── check_cluster.sh # Connectivity test for cluster nodes
│   └── utils/          # Utility scripts
├── templates/           # Starter code/config templates
│   ├── CMakeLists.txt.template
│   ├── template.c.template
│   └── template.cu.template
└── docs/               # Additional documentation, notes
    └── HPC-tips.md    # HPC debugging & performance tips
```

## 5. Current Implementation Status
### Homeworks
- **HW1 (MPI Point-to-Point Matrix Multiplication):** Completed, tested with local cluster or oversubscribing ranks. Verified partial speedups and correctness for matrix multiplication with 1–8 processes.
- **HW2 (MPI Collective Communication):** In progress, focusing on scatter/gather and reduce operations for advanced HPC patterns.
- **HW3+** (Not all enumerated here)...

### Final Project: AlexNet Inference
- **Design:** Data-parallel approach (each MPI rank holds a copy of the model weights, processes a slice of input data).
- **Layers Implemented:**
  1. **Conv1 → ReLU1 → Pool1**
  2. **Conv2 → ReLU2 → Pool2 → LRN2**
- **Naive Kernels:** Basic convolution, max-pooling, and LRN. Next steps include adding more layers and eventually fully connected + softmax.
- **Testing Scripts:** We have an extended script that runs the final project with different process counts and synthetic data multipliers, verifying correctness and capturing kernel timing.

## 6. Development Workflow
1. **Clone & Setup:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   # optional: if using WSL2
   sudo apt update && sudo apt install build-essential openmpi-bin libopenmpi-dev
   ```

2. **Scaffold a Homework:**
   ```bash
   bash scripts/scaffold_hw.sh 4  # Creates homeworks/hw4 with skeleton
   ```

3. **Implement / Update Code:**
   - For homework: Edit in `homeworks/hwX/src/template.cu` or `.c`
   - For final project: Edit in `final_project/src/`

4. **Test Locally:**
   ```bash
   bash scripts/test_hw.sh X
   # or for final project:
   cd final_project
   make clean && make
   mpirun --oversubscribe -np 2 ./template
   ```

5. **Package for Submission:**
   ```bash
   bash scripts/package_hw.sh X LastName FirstName
   ```

## 7. Automation Scripts
- `scaffold_hw.sh <hw_num>`: Creates homework structure with templates
- `test_hw.sh <hw_num>`: Builds and tests homework
- `package_hw.sh <hw_num> <last> <first>`: Packages for submission
- `run_hw.sh <hw_num> <last> <first>`: Combines test and package
- Final project scripts in `final_project/scripts/` handle testing and results analysis

## 8. Build & Test Instructions
### Single Homework
```bash
bash scripts/test_hw.sh 2
bash scripts/package_hw.sh 2 Doe John
```

### Final Project
```bash
cd final_project
make clean && make
bash scripts/test_final.sh
bash scripts/summarize_results.sh
```

## 9. Submission Guidelines
- Submit as `hw<num>-lastname-firstname.tgz`
- Include only `template.c/cu` and `Makefile`
- Must compile on Fedora 37 with GCC 12 and CUDA 12.x
- No late submissions accepted

## 10. Troubleshooting
Common issues:
- Makefile TAB vs. Spaces
- MPI header not found
- CUDA compiler errors
- Timeouts in test scripts
- Linker errors

## 11. Future Directions
- Expanded AlexNet layers
- Model parallelism
- Performance optimization
- Distributed training
- Advanced profiling integration

## 12. References & Resources
- MPI Forum: mpi-forum.org
- NVIDIA CUDA Docs: docs.nvidia.com/cuda/
- PMPP Book (4th Ed.)
- HPC Community Channels
