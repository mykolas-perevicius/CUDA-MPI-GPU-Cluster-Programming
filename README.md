
## Setup and Configuration

1.  **Clone the repository:**
    `git clone <your-repo-url>`
    `cd cs485-gpu-cluster-programming`
2.  **Configure Cluster Details:**
    *   Copy the template: `cp config/cluster.conf.template config/cluster.conf`
    *   Edit `config/cluster.conf` with your actual node hostnames/IPs and username (initially for PC/Laptop, later for the two course laptops).
    *   **Note:** `config/cluster.conf` is ignored by git.

## Development Workflow

1.  **Scaffold New Homework:**
    *   Run `bash scripts/scaffold_hw.sh <hw_number>` (e.g., `bash scripts/scaffold_hw.sh 1`).
    *   This creates the `homeworks/hw<number>` directory with starter files (`src/template.c`, `Makefile`/`CMakeLists.txt`, `summary.md`).
2.  **Implement Code:**
    *   Edit the source file(s) in `homeworks/hw<number>/src/`.
3.  **Test Locally:**
    *   Run `bash scripts/test_hw.sh <hw_number>`.
    *   This script will compile the code (using Make or CMake) and run it locally using `mpirun` with various process counts (`-np 1` to `8`) and problem sizes, simulating the professor's test script.
4.  **Generate Summary (AI Integration):**
    *   Use an AI assistant (like ChatGPT, Copilot) to help populate `homeworks/hw<number>/summary.md`.
    *   Provide your code, relevant book chapters/notes, and ask for explanations, potential pitfalls, performance tips, and sample exam questions.
    *   **Example Prompt Context:** "I'm working on HW{X} for CS485 (GPU Cluster Programming) covering {topic}. The goal is {problem description}. Here's my code: [code]. Relevant resources: {list resources}. Please generate a summary covering key concepts, MPI/CUDA functions used, potential issues, and 3 potential final exam questions with answers."
5.  **Package for Submission:**
    *   Once tests pass, run `bash scripts/package_hw.sh <hw_number> <your_lastname> <your_firstname>`.
    *   This creates the required `hw<number>-lastname-firstname.tgz` archive in the root directory, containing only the necessary files (`template.c`/`cu`, `Makefile`).
6.  **Combined Test & Package:**
    *   Run `bash scripts/run_hw.sh <hw_number> <your_lastname> <your_firstname>`.
    *   This runs the tests first, and only if they succeed, it proceeds to package the assignment.

## Automation Scripts

*   **`scripts/scaffold_hw.sh <hw_num>`:** Sets up directory structure for homework `hw_num`.
*   **`scripts/test_hw.sh <hw_num>`:** Compiles and runs tests for `hw_num` locally, simulating multiple processes and problem sizes.
*   **`scripts/package_hw.sh <hw_num> <last> <first>`:** Creates `hw<num>-last-first.tgz` for submission. **Crucially uses the submission `Makefile`, not necessarily the CMake build system.**
*   **`scripts/run_hw.sh <hw_num> <last> <first>`:** Runs tests, then packages if tests pass.
*   **`scripts/check_cluster.sh`:** (For later use) Reads `config/cluster.conf` and performs basic connectivity tests (ping, ssh, mpirun hostname) between configured nodes.

## Build System

*   **Homeworks 1-3 (MPI Only):** Primarily use the provided `Makefile`.
*   **Homeworks 4-9 (MPI + CUDA):** Use `CMakeLists.txt` for development (handles finding MPI and CUDA libraries). A separate, simpler `Makefile` (potentially copied from `templates/Makefile.template` and adjusted) will be included in the `homeworks/hwX` directory *specifically for packaging and submission* to meet course requirements.
*   The `test_hw.sh` script detects whether to use `make` or `cmake/make` for building during local tests.

## Final Environment Target

While development starts in WSL2 for convenience, **all code must ultimately compile and run correctly in the specified Fedora 37 environment with GCC 12 and CUDA Toolkit 12.x**. The automated testing script aims to catch issues early, but final validation should happen on a system matching the grading environment (e.g., the provided laptops).

## Submission

*   Use the `package_hw.sh` or `run_hw.sh` script to generate the `.tgz` file.
*   Upload the generated `.tgz` file to Canvas by the deadline.
*   **Strict Adherence:** Ensure correct directory (`hwX-lastname-firstname`) and executable (`template`) naming within the archive, as grading is automated.