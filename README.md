
```markdown
# CS485: GPU Cluster Programming (MPI+CUDA) - Comprehensive Repository

Welcome to the official repository for CS485: GPU Cluster Programming at NJIT. This repo contains all course-related materials—homework assignments, the final project code (including our AlexNet inference implementation), automation scripts, configuration files, and supporting documentation. Our objective is to develop robust, high-performance parallel programs using MPI, CUDA, and their combination for GPU clusters.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
  - [Project Directories](#project-directories)
  - [Automation Scripts](#automation-scripts)
  - [Configuration & Templates](#configuration--templates)
- [Environment & Prerequisites](#environment--prerequisites)
- [Development Workflow](#development-workflow)
  - [Homeworks](#homeworks)
  - [Final Project (AlexNet Inference)](#final-project-alexnet-inference)
  - [AI Integration & Documentation](#ai-integration--documentation)
- [Build and Test Systems](#build-and-test-systems)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [Contribution & Version Control](#contribution--version-control)
- [Final Environment & Submission Guidelines](#final-environment--submission-guidelines)
- [Contact and Support](#contact-and-support)

---

## Overview

This repository serves as a comprehensive workspace for CS485, where you will complete assignments that progressively build your skills in MPI, CUDA, and ultimately MPI+CUDA for GPU cluster programming. The repository is designed for robust local development (on WSL2 under Ubuntu) and compatibility with the strict Fedora 37 target environment. Our final project, which demonstrates an implementation of AlexNet inference (with the first block: Conv1 → ReLU → MaxPool1 and the second block: Conv2 → ReLU → MaxPool2 → LRN2), leverages a data-parallel strategy over MPI ranks. Automation scripts ensure reproducible builds, testing across various configurations, and final packaging for submission.

---

## Repository Structure

The repository is structured to clearly separate homework assignments, final project code, automation scripts, configuration files, and supporting documentation. Below is an overview of the key directories and files:

```
.
├── .github/                    
│   └── workflows/              # GitHub Actions CI configurations (optional)
├── .gitignore                  # Defines ignored files and directories
├── .vscode/                    
│   ├── c_cpp_properties.json   # VS Code IntelliSense settings (MPI and CUDA paths)
│   └── settings.json           # VS Code settings (e.g., Makefile TAB settings, CMake options)
├── README.md                   # This comprehensive repository overview
├── config/                     
│   └── cluster.conf.template   # Template for cluster node configuration (IPs, usernames, etc.)
├── homeworks/                  
│   └── hwX/                    # Homework assignments (X=1,2,...,9)
│       ├── src/                # Homework source files (.c or .cu)
│       ├── Makefile            # Generated Makefile for submission (ensuring correct TAB usage)
│       ├── CMakeLists.txt      # CMake build configuration (for homework assignments HW4+)
│       ├── build/              # Build directory created by test scripts (ignored by Git)
│       └── summary.md          # AI-generated/manual summaries, notes, and exam prep material
├── final_project/              
│   ├── Makefile                # Unified Makefile to build the final MPI+CUDA project (using nvcc -ccbin=mpicxx)
│   ├── include/                # Header files for final project (e.g., alexnet.hpp, layers.hpp, mpi_helper.hpp)
│   └── src/                    # Final project source code (main.cpp, alexnet_hybrid.cu, layers.cu)
├── scripts/                    
│   ├── scaffold_hw.sh          # Automates the creation of homework/project directories and files
│   ├── test_hw.sh              # Compiles and runs tests locally for homework assignments, simulating cluster runs
│   ├── package_hw.sh           # Packages a homework assignment into the required .tgz archive for submission
│   ├── run_hw.sh               # Runs test_hw.sh and, if successful, calls package_hw.sh automatically
│   ├── check_cluster.sh        # Performs basic connectivity and MPI sanity checks on configured cluster nodes
│   └── run_final_project_extended.sh  # Automates building, running, and logging tests for the final project
└── templates/                  
    ├── CMakeLists.txt.template # Starter CMake build files for homework assignments
    ├── template.c.template     # C source template for homework assignments
    └── template.cu.template    # CUDA source template for homework assignments (HW4+)
```

### Project Directories

- **homeworks/**  
  Contains separate subdirectories (hw1, hw2, …, hw9) for each homework assignment. Each contains a complete mini-project with build files, source code, and an accompanying summary for self-review and exam study.

- **final_project/**  
  Contains the complete final project code for AlexNet Inference using MPI+CUDA. Here, the code is organized into:
  - **include/** – Header files for the project (defines layer parameters, API prototypes, MPI helpers, etc.).
  - **src/** – Source files including `main.cpp` (handles MPI initialization, input generation, parameter broadcast, and output gathering), `alexnet_hybrid.cu` (defines the unified forward pass across two blocks), and `layers.cu` (naive CUDA kernel implementations for convolution, ReLU, max pooling, and LRN).

### Automation Scripts

The **scripts/** directory provides Bash scripts that streamline your workflow:
- **scaffold_hw.sh:** Sets up new homework/project directories with boilerplate files (Makefile, source template, summary.md).
- **test_hw.sh:** Compiles the code (using make for MPI-only homework or CMake for MPI+CUDA homework) and runs tests with `mpirun` over different process counts and input sizes (with timeout handling).
- **package_hw.sh:** Packages homework submissions into a standardized .tgz archive following strict naming conventions.
- **run_hw.sh:** Combines testing and packaging in a single step.
- **check_cluster.sh:** Verifies network connectivity, SSH access, and MPI functionality between nodes in your cluster.
- **run_final_project_extended.sh:** Specifically for the final project, this script builds the executable, runs extended tests over varying MPI ranks and synthetic data multipliers, and logs performance/timing output for review.

### Configuration & Templates

- **config/**  
  Contains configuration templates (like `cluster.conf.template`). Customize these for your deployment or cluster environment.  
- **templates/**  
  Contains starter files that are used by the scaffolding scripts for creating new homework or project directories.

---

## Environment & Prerequisites

### Development Environment (WSL2 Recommended)

For ease of development, we recommend using WSL2 on Windows (Ubuntu is preferred). However, ensure that your code is ultimately compatible with the Fedora 37 target environment specified by the course.

**Prerequisites:**

1. **WSL2/Ubuntu Setup:**  
   Ensure WSL2 is installed on your Windows machine.
   
2. **Linux Tools:**  
   ```bash
   sudo apt update
   sudo apt install build-essential cmake git python3 python3-pip ninja-build gcc-12 g++-12
   ```
   (GCC 12/G++ 12 are required to match the target compiler.)

3. **Open MPI:**  
   ```bash
   sudo apt install openmpi-bin libopenmpi-dev
   ```

4. **NVIDIA CUDA Toolkit for WSL2:**  
   - Verify that your Windows NVIDIA driver is installed and compatible with CUDA 12.x.  
   - Follow the official [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).  
   - Example installation for CUDA Toolkit 12.4:
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
     sudo dpkg -i cuda-keyring_1.1-1_all.deb
     sudo apt-get update
     sudo apt-get install -y cuda-toolkit-12-4
     echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
     echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
     source ~/.bashrc
     ```
   - Validate with: `nvcc --version` and `nvidia-smi`.

### Target Environment for Grading

- **OS:** Fedora 37  
- **Compiler:** GCC 12  
- **CUDA Toolkit:** 12.x  
- **MPI:** OpenMPI (with CUDA support)  
*Make sure to test your code on similar hardware (or the provided course laptops) before final submission.*

---

## Development Workflow

Our workflow is designed to maximize automation while ensuring that code development, testing, and documentation are integrated seamlessly.

### Homeworks

1. **Scaffold New Homework:**  
   Use the provided scaffold script:
   ```bash
   bash scripts/scaffold_hw.sh <hw_number>
   ```
   This generates a directory (e.g., `homeworks/hw1/`) with a source template, Makefile, and placeholder summary file.

2. **Develop Code:**  
   Edit the source files in `homeworks/hwX/src/` according to homework requirements (C for MPI-only assignments; CUDA/C++ for MPI+CUDA assignments).

3. **Local Testing:**  
   Run:
   ```bash
   bash scripts/test_hw.sh <hw_number>
   ```
   to build and test the homework, simulating different MPI process counts and input sizes with a 30-second timeout on each test run.

4. **Summary & AI Integration:**  
   Use AI tools to generate and update the `summary.md` file with explanations, debugging tips, and potential exam questions related to the homework topics.

5. **Submission Packaging:**  
   Package the homework using:
   ```bash
   bash scripts/package_hw.sh <hw_number> <your_lastname> <your_firstname>
   ```
   or run the combined script:
   ```bash
   bash scripts/run_hw.sh <hw_number> <your_lastname> <your_firstname>
   ```
   This ensures that only the required files (source and submission Makefile) are archived into a `.tgz` file following the naming convention.

### Final Project: AlexNet Inference

1. **Project Development:**  
   - The final project resides in the `final_project/` directory.  
   - Source code is organized under `final_project/src/` and `final_project/include/`.
   - The project implements AlexNet inference in two blocks:  
     * Block 1: Conv1 → ReLU → MaxPool1 (already completed)  
     * Block 2: Conv2 → ReLU → MaxPool2 → LRN2 (newly added)
   - Weight initialization for both Conv1 and Conv2 is performed on rank 0; weights are broadcast to ensure consistency across MPI ranks.

2. **Build and Execute:**  
   - A unified Makefile in `final_project/Makefile` builds the executable (named `template`) using `nvcc -ccbin=mpicxx`.  
   - Use MPI to run tests with varying rank counts. For example:
     ```bash
     mpirun --oversubscribe -np 4 ./final_project/template [multiplier]
     ```
   - The `alexnetForward` function combines CUDA kernel calls (for convolution, activation, pooling, and LRN) and measures execution time.

3. **Extended Testing & Logging:**  
   - Execute:
     ```bash
     bash scripts/run_final_project_extended.sh
     ```
     to build, run tests with multiple MPI process counts and synthetic data multipliers, and automatically generate a performance log summary.

### AI Integration & Documentation

- **Summaries & Cheat Sheets:**  
  Use AI services to generate detailed summaries of each homework and project module. These summaries (located in each `summary.md`) provide:
  - A high-level explanation of the problem  
  - Descriptions of key MPI/CUDA functions and their roles  
  - Debugging tips and common pitfalls  
  - Performance considerations and potential exam questions with answers

- **Prompt Examples:**  
  For instance, you might instruct:  
  _"I have implemented the AlexNet Conv2 block with MPI+CUDA. Please summarize the design decisions, describe the MPI broadcast, and provide three potential exam questions on how to optimize CUDA kernels."_

- **Documentation:**  
  This README and in-code comments form the backbone of our project documentation, ensuring that both human reviewers (e.g., the professor) and automated grading scripts can understand our design choices.

---

## Build and Test Systems

### Build System

- **Final Project:**  
  The final project is built using a single Makefile (in `final_project/Makefile`) which compiles with:
  ```makefile
  nvcc -ccbin=mpicxx ... -o template
  ```
- **Homework Assignments:**  
  Homework assignments may use either a generated Makefile (for MPI-only assignments) or a combination of CMake and Make (for MPI+CUDA assignments) located in the respective homework directories.

### Testing Automation

- **Local Testing:**  
  Run the testing scripts (`test_hw.sh` or `run_final_project_extended.sh`) to execute your code with a variety of configurations (process counts, input sizes). Each run has a 30-second timeout; logs are saved in `final_project/logs_extended/`.
- **Output Verification:**  
  The testing scripts extract critical timing and performance information (e.g., kernel execution times, MPI_Wtime measurements) and print concise summaries for quick review.
- **Cluster Simulation:**  
  Although local tests are run on one machine (with simulated MPI oversubscription), the provided `check_cluster.sh` script can be used to verify connectivity between multiple nodes when a true cluster is available.

---

## Troubleshooting & Common Issues

- **Makefile Errors:**  
  Missing TAB characters can result in errors like `missing separator`.  
  *Solution:* Ensure your editor uses TAB characters for Makefiles (as configured in `.vscode/settings.json`).

- **Include Path Issues:**  
  Errors such as `cannot open source file "mpi.h"` or `"cuda_runtime.h"` are usually due to misconfigured include paths.  
  *Solution:* Verify that `.vscode/c_cpp_properties.json` contains correct paths (use `mpicc --showme:incdirs` for MPI and check `/usr/local/cuda-12.x/include` for CUDA).

- **MPI Communication Errors:**  
  If your MPI-based executions fail (e.g., timeout or connectivity issues), run `scripts/check_cluster.sh` to validate network, SSH, and MPI configurations.
  
- **CUDA Kernel Failures:**  
  Use CUDA error checking macros (e.g., `CUDA_CHECK`) to report any runtime errors in kernel launches. Check device memory allocations and kernel grid/block configurations if you encounter unexpected results.

- **Build Environment Mismatch:**  
  Ensure that your final tests are conducted on an environment that mimics Fedora 37 with GCC 12 and the correct CUDA Toolkit version. Develop on WSL2, but test on the target system when possible.

---

## Contribution & Version Control

* **Commit Frequently:**  
  Save each incremental change—whether fixing a kernel bug or updating a test script—with descriptive commit messages.
  ```bash
  git add .
  git commit -m "Implemented Conv2 and LRN2 kernels for AlexNet Block2"
  ```
* **Branching:**  
  Use feature branches for major new functionality or experimental changes. Merge only after thorough testing.
* **Code Reviews:**  
  Review commits and incorporate code comments to help maintain a high standard of clarity and maintainability.
* **Remote Backups:**  
  Regularly push commits to GitHub (or your central repository) for backup and continuous integration (CI) purposes.

---

## Final Environment & Submission Guidelines

1. **Environment Target:**  
   Final submissions must compile and run correctly on Fedora 37, using GCC 12 and the CUDA Toolkit 12.x.  
2. **Packaging:**  
   Use `scripts/package_hw.sh` or `scripts/run_hw.sh` to create the final `.tgz` file (or a similar archive) that contains only the required source files and submission Makefile. Ensure the archive adheres to the naming convention (e.g., `hw1-lastname-firstname.tgz` for homework or `finalproject-lastname-firstname.tgz` for the final project).
3. **Submission:**  
   Upload the single `.tgz` file to Canvas. Double-check all file names and directory structures.
4. **Grading Script Expectations:**  
   The instructor’s script will untar, compile, and run your submission with varying process counts and problem sizes. Ensure that your code is robust against invalid inputs and time constraints.

---

## Contact and Support

If you encounter issues or have questions during development:

* **Course Instructor/TA:** Refer to your course webpage or contact your TA for technical help related to cluster or MPI setup.
* **GitHub Issues:** Use the repository’s issue tracker to report bugs, request new features, or ask for clarifications.
* **Community Resources:** Check out documentation on [NVIDIA Developer](https://developer.nvidia.com/), [MPI Forum](https://mpi-forum.org/), and related forums or Stack Overflow.

Happy coding—and good luck with your high-performance computing journey in CS485!

```
