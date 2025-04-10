# CS485: GPU Cluster Programming (MPI+CUDA) - Project Workspace

This repository contains homework assignments, final project code, automation scripts, testing frameworks, and resources for the CS485 GPU Cluster Programming course at NJIT. The objective is to develop, validate, and optimize MPI, CUDA, and MPI+CUDA applications, ensuring compatibility with the required Fedora 37 environment.

**Version:** 1.2 (Updated with AlexNet Implementation and extended testing scripts)

---

## Course Information

- **Course:** CS485 Selected Topics: GPU Cluster Programming
- **Instructor:** Andrew Sohn
- **Topics:** MPI, CUDA, GPU Cluster Programming, Parallel Architectures
- **Key Technologies:** C, C++, CUDA, MPI, Linux (Fedora 37), Bash, Make
- **Course Webpage:** [http://web.njit.edu/~sohna/cs485](http://web.njit.edu/~sohna/cs485)
- **Textbooks:**
  - MPI: A Message Passing Interface Standard v3.1 (Free)
  - Programming Massively Parallel Processors (PMPP), 4th Ed.
- **Required Grading Environment:** Fedora 37, GCC 12, CUDA Toolkit 12.x

---

## Project Goals

1. Complete all assignments with correctness and optimal performance.
2. Implement and benchmark AlexNet inference using MPI+CUDA, demonstrating significant parallel speedup.
3. Automate build, test, and packaging workflows using scripts.
4. Maintain clean, well-documented, and organized code.
5. Utilize AI assistance effectively for development, debugging, and conceptual clarity.

---

## Current Implementation Status

### AlexNet Inference (MPI+CUDA)
- **Implemented Layers:** Conv1 → ReLU1 → MaxPool1 → Conv2 → ReLU2 → MaxPool2 → LRN2
- **Data Parallelism:** Each MPI rank processes distinct input data; full model weights are broadcast using MPI.
- **CUDA Kernels:** Naive implementation provided for clarity; optimization planned.
- **Automated Testing:** Comprehensive testing script available; runs with varying MPI ranks and data sizes, reporting execution times and correctness.

---

## Repository Structure

```
.
├── README.md
├── final_project/
│   ├── include/         # Header files (layers.hpp, alexnet.hpp)
│   ├── src/             # Source files (main.cpp, alexnet_hybrid.cu, layers.cu)
│   ├── Makefile         # Builds project executables
│   └── scripts/
│       ├── test_final.sh   # Automated tests with MPI and CUDA
│       └── summarize_results.sh # Summarizes test outputs
├── homeworks/
├── scripts/             # General automation scripts
└── docs/
    └── notes.md         # Development notes and insights
```

---

## How to Build & Test

### Building the Project

Navigate to the `final_project` directory:

```bash
cd final_project
make
```

This will compile both the main executable (`template`) and convolution test executable (`conv_test`).

### Running Automated Tests

Execute the automated test script from within `final_project`:

```bash
bash scripts/test_final.sh
```

This runs a series of MPI-based tests with varying ranks and dataset multipliers, verifying correct MPI communication, CUDA kernel correctness, and timing performance.

### Summarizing Test Results

To summarize testing results into an easily readable format, run:

```bash
bash scripts/summarize_results.sh
```

---

## Development Environment

Recommended to use WSL2 with Ubuntu for initial development and testing. Final validation must be on Fedora 37 with GCC 12 and CUDA 12.x.

Install necessary tools:

```bash
sudo apt update
sudo apt install build-essential cmake git openmpi-bin libopenmpi-dev cuda-toolkit-12-4
```

Ensure your CUDA and MPI installations are correctly configured.

---

## AI Assistance Usage

AI tools have been extensively used for:
- Generating clear, detailed documentation and code comments
- Assisting in debugging and resolving CUDA and MPI errors
- Automating summaries of testing and implementation steps
- Enhancing learning by generating conceptual explanations

---

## Troubleshooting

Common troubleshooting steps:
- Ensure environment variables (`PATH`, `LD_LIBRARY_PATH`) are correctly set.
- Verify MPI functionality (`mpirun --version`, `mpirun -np 2 hostname`).
- Confirm CUDA installation (`nvcc --version`, `nvidia-smi`).
- Regularly pull updates and check repository status with `git status`.

---

## Future Directions

- Implement optimized CUDA kernels leveraging shared memory and tiling.
- Add further AlexNet layers (fully connected, softmax classification).
- Extend automated scripts to handle dynamic MPI rank discovery and GPU load balancing.

---

## Final Notes

Please ensure thorough testing on Fedora 37 prior to submission. Consistent documentation and robust automation scripts are critical for reproducibility and grading clarity.

