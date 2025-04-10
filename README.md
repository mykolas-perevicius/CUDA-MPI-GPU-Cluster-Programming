# CS485: GPU Cluster Programming (MPI+CUDA) - Project Workspace

This repository contains the homework assignments, project code, automation scripts, and learning materials for the CS485 course at NJIT. The goal is to develop and test MPI, CUDA, and combined MPI+CUDA programs, automating the workflow to ensure consistency and prepare for submission.

**Version:** 1.1 (Updated with robust Makefile generation and workflow clarifications)

## Course Information

*   **Course:** CS485 Selected Topics: GPU Cluster Programming
*   **Instructor:** Andrew Sohn
*   **Topics:** MPI (Point-to-Point, Collective, One-Sided), CUDA (Architecture, Memory, Kernels, Patterns), GPU Cluster Programming (MIMD, SIMD, SPMD)
*   **Key Technologies:** C, C++, CUDA, MPI (Open MPI), Linux (Fedora 37 Target), Bash, CMake, Make
*   **Course Webpage:** [http://web.njit.edu/~sohna/cs485](http://web.njit.edu/~sohna/cs485)
*   **Textbooks:**
    *   MPI: A Message Passing Interface Standard v3.1 (Free)
    *   Programming Massively Parallel Processors (PMPP), 4th Ed.
*   **Required Grading Environment:** Fedora 37, GCC 12, CUDA Toolkit 12.x (Strict requirement for automated grading)

## Project Goals

1.  Successfully complete all homework assignments (MPI, CUDA, MPI+CUDA).
2.  Develop a final project demonstrating significant parallel speedup using MPI+CUDA.
3.  Automate the build, test, and packaging process using Bash scripts and appropriate build tools (Make/CMake).
4.  Maintain clear, well-documented code and project structure.
5.  Utilize AI assistance (like this chat) for generating summaries, explanations, debugging help, and potential exam questions to reinforce learning.
6.  Ensure code compatibility with the specified Fedora 37 grading environment.

## Prerequisites (Development Environment - WSL2 Recommended)

This setup assumes development within WSL2 (Ubuntu recommended) for convenience. Final testing should ideally occur on a Fedora 37 system.

1.  **WSL2:** Installed and functioning on Windows.
2.  **Linux Distribution:** Ubuntu (or similar Debian-based) inside WSL2.
3.  **Essential Build Tools:**
    ```bash
    sudo apt update
    sudo apt install build-essential cmake git python3 python3-pip ninja-build gcc-12 g++-12
    ```
    *(Note: `gcc-12`/`g++-12` are needed to match the course's target compiler, even if your default GCC is newer. Ninja is used by CMake. Python is optional but useful.)*
4.  **Open MPI:**
    ```bash
    sudo apt install openmpi-bin libopenmpi-dev
    ```
5.  **NVIDIA CUDA Toolkit for WSL2:**
    *   Ensure your Windows NVIDIA **host driver** is installed and compatible with the target CUDA Toolkit version (e.g., 12.x required by the course). Check driver release notes.
    *   Follow the official [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) for installation within your WSL distribution.
    *   **Install the specific CUDA Toolkit version required by the course (e.g., 12.2, 12.4).** Example for 12.4:
        ```bash
        # Add NVIDIA repo key (if not already done)
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        # Install the toolkit
        sudo apt-get install -y cuda-toolkit-12-4
        # Add to PATH/LD_LIBRARY_PATH (adjust version number!)
        echo '# Add CUDA paths' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
        source ~/.bashrc
        ```
    *   Verify installation: `nvcc --version` and `nvidia-smi` (within WSL).

## Repository Structure

.
├── .github/workflows/ # GitHub Actions CI (Optional)
├── .gitignore
├── .vscode/ # VS Code specific settings
│ ├── c_cpp_properties.json # IntelliSense configuration (MPI/CUDA paths)
│ └── settings.json # VS Code editor/extension settings
├── README.md # This file
├── config/
│ └── cluster.conf.template # Template for cluster node configuration (IPs, user)
├── homeworks/ # Root directory for all homework assignments
│ └── hwX/ # Specific homework directory (e.g., hw1)
│ ├── src/ # Source code (template.c or template.cu)
│ ├── Makefile # Makefile for building (Generated, REQUIRED for submission)
│ ├── CMakeLists.txt # CMake build file (Generated for HW4+, dev/test use only)
│ ├── build/ # Build directory created by test script for CMake builds (.gitignore'd)
│ └── summary.md # AI-generated/manual summary, notes, exam questions
├── scripts/ # Automation scripts
│ ├── scaffold_hw.sh # Creates structure & initial files for a new homework
│ ├── test_hw.sh # Compiles & runs tests locally (simulating cluster runs)
│ ├── package_hw.sh # Packages homework directory into .tgz for submission
│ ├── run_hw.sh # Runs test_hw.sh then package_hw.sh if tests pass/timeout
│ ├── check_cluster.sh # Basic cluster connectivity tests (ping, ssh, mpi)
│ └── utils/ # Utility/one-off scripts (e.g., restructuring)
└── templates/ # Starter code/config templates (except Makefile)
├── CMakeLists.txt.template
├── template.c.template
└── template.cu.template


## Setup and Configuration

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-folder-name>
    ```
2.  **Install Prerequisites:** Follow the steps in the "Prerequisites" section above.
3.  **Configure Cluster Details (for `check_cluster.sh`):**
    *   Copy the template: `cp config/cluster.conf.template config/cluster.conf`
    *   Edit `config/cluster.conf` with actual hostnames/IPs and username for your development cluster (e.g., PC + Laptop) or the final course laptops.
    *   *Note:* `config/cluster.conf` is ignored by git.

## Development Workflow

1.  **Scaffold New Homework:**
    *   `bash scripts/scaffold_hw.sh <hw_number>` (e.g., `bash scripts/scaffold_hw.sh 2`)
    *   This creates `homeworks/hw<number>/` with `src/`, `summary.md`, the submission `Makefile`, and `CMakeLists.txt` (for HW4+). It copies the appropriate source template (`.c` or `.cu`) into `src/`.
2.  **Implement Code:**
    *   Edit the source file(s) in `homeworks/hw<number>/src/`.
3.  **Test Locally:**
    *   `bash scripts/test_hw.sh <hw_number>`
    *   This script:
        *   Changes into the `homeworks/hw<number>` directory.
        *   Builds the code: Uses `make` for HW1-3, uses `cmake` & `make` (within `build/`) for HW4+.
        *   Runs the compiled `template` executable using `mpirun` across a range of process counts (1-8) and problem sizes (128-2048).
        *   Applies a 30-second timeout to each `mpirun` command.
        *   Reports overall status: PASSED, FAILED, or INCONCLUSIVE (if timeouts occurred).
4.  **Generate Summary / Review Concepts (AI Integration):**
    *   Use an AI assistant (like this chat) to populate `homeworks/hw<number>/summary.md`.
    *   **Prompt Idea:** "I've completed HW{X} ({Topic}) for CS485. Here's the code from `src/template.c`: [paste code]. Key concepts are {list concepts}. Please generate a summary for `summary.md` including: 1. Brief Problem Explanation, 2. Key MPI/CUDA functions used and their roles, 3. Common Pitfalls/Debugging Tips for this type of problem, 4. Performance Considerations, 5. 3 Potential Final Exam Questions & Answers related to this."
5.  **Package for Submission:**
    *   `bash scripts/package_hw.sh <hw_number> <your_lastname> <your_firstname>`
    *   **Important:** This script *only* includes the source file (`template.c` or `.cu`) and the generated `Makefile` (from the homework root, *not* from a `build/` directory) in the required `hw<number>-lastname-firstname/` structure before creating the `.tgz` archive. Your lastname/firstname will be converted to lowercase.
6.  **Combined Test & Package:**
    *   `bash scripts/run_hw.sh <hw_number> <your_lastname> <your_firstname>`
    *   Runs `test_hw.sh`. If tests pass (exit code 0) or are inconclusive due to timeout (exit code 2), it proceeds to run `package_hw.sh`. If tests fail (exit code 1), packaging is skipped.

## Automation Scripts Explained

*   **`scripts/scaffold_hw.sh <hw_num>`:** Creates the standard directory structure and initial files for homework `hw_num`, including generating the required submission `Makefile` directly to avoid tab issues.
*   **`scripts/test_hw.sh <hw_num>`:** The primary testing script. Selects the build method (Make/CMake) based on `hw_num`, compiles, runs `mpirun` with multiple configurations and a timeout, reports status via exit code (0=Pass, 1=Fail, 2=Timeout).
*   **`scripts/package_hw.sh <hw_num> <last> <first>`:** Creates the `.tgz` submission archive (`hw<num>-lastname-firstname.tgz`) containing only the required source file and root `Makefile`, matching the specified naming convention.
*   **`scripts/run_hw.sh <hw_num> <last> <first>`:** Convenience script to run tests and then package if tests didn't explicitly fail.
*   **`scripts/check_cluster.sh`:** Reads `config/cluster.conf` to perform basic connectivity tests (ping, passwordless ssh, basic `mpirun hostname`) between configured nodes. Useful before running actual distributed jobs.

## Build System Strategy

*   **HW1-3 (MPI Only):** The `Makefile` generated by `scaffold_hw.sh` is used for both testing (`test_hw.sh` calls `make`) and submission packaging (`package_hw.sh` includes this `Makefile`).
*   **HW4-9 (MPI + CUDA):**
    *   **Development/Testing:** `test_hw.sh` uses the `CMakeLists.txt` (copied from template) to configure and build (using `cmake` and `make`) inside a temporary `build/` directory within the homework folder. This handles finding MPI and CUDA libraries more easily.
    *   **Submission:** `package_hw.sh` ignores the CMake system and the `build/` directory. It packages the source (`.cu`) file along with the root `Makefile` (which was *also* generated by `scaffold_hw.sh` and should contain basic rules for compiling the `.cu` file with `nvcc` and linking with `mpicc`/`mpiCC` for the grading environment). You may need to manually adjust this submission `Makefile` for specific CUDA library linking if required by later assignments.

## VS Code Configuration

*   **`.vscode/c_cpp_properties.json`:** Configures IntelliSense for the C/C++ extension. **Crucial:** Ensure the `includePath` contains the correct paths to your OpenMPI and CUDA include directories for proper code completion and error checking.
*   **`.vscode/settings.json`:**
    *   `"cmake.configureOnOpen": false` is recommended to prevent VS Code from interfering with the script-based build process.
    *   `"[makefile]": { "editor.insertSpaces": false }` helps prevent accidental conversion of TABs to spaces if you manually edit Makefiles.

## Troubleshooting Common Issues

*   **`Makefile:X: *** missing separator. Stop.`:** The command line on line X (or nearby) is missing a required leading **TAB** character. It likely has spaces instead. The `scaffold_hw.sh` script *should* prevent this, but manual edits can reintroduce it. Use an editor that shows whitespace or configure VS Code (see above) for Makefiles.
*   **`cannot open source file "mpi.h"` (in VS Code):** IntelliSense error. Verify the `includePath` in `.vscode/c_cpp_properties.json` points to the correct OpenMPI include directory (find using `mpicc --showme:incdirs`). Reload VS Code (`Developer: Reload Window`) after editing.
*   **`cannot open source file "cuda_runtime.h"` (in VS Code):** IntelliSense error. Verify the `includePath` in `.vscode/c_cpp_properties.json` points to the correct CUDA include directory (e.g., `/usr/local/cuda-12.X/include`). Reload VS Code.
*   **CMake configuration errors:** Ensure `cmake`, `gcc-12`, `g++-12` are installed. Check paths in `CMakeLists.txt`. Run `rm -rf build` in the specific homework directory and let `test_hw.sh` regenerate it.
*   **`mpirun` errors:** Check MPI installation (`mpirun --version`), PATH variables, firewall settings (less common locally), or issues within your MPI code itself. The `check_cluster.sh` script can help verify basic MPI functionality between nodes later.

## Final Environment Target & Submission

*   **Goal:** While WSL2 is convenient for development, your final code **must** compile and run correctly on **Fedora 37 with GCC 12 and the specified CUDA Toolkit 12.x version**. Test on the provided course laptops if possible before final submission.
*   **Packaging:** Always use `scripts/package_hw.sh` or `scripts/run_hw.sh` to create the final `.tgz` file.
*   **Submission:** Upload the single `.tgz` file to Canvas. Double-check the naming convention. The automated grading is strict.

## Using Git Effectively

*   **Commit Often:** Save your progress frequently! After getting a piece of functionality working or finishing a work session:
    ```bash
    git add . # Stage all changes (or specify files)
    git commit -m "Implemented MPI_Send/Recv for matrix A in HW1"
    ```
*   **Check Status:** See what files are changed: `git status`
*   **View History:** See past commits: `git log`
*   **Push Backups:** Regularly push your local commits to your remote GitHub repository: `git push origin main` (or your branch name). This is your offsite backup!