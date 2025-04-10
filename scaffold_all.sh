#!/bin/bash
# scaffold_all.sh
# Unified repository scaffolding script for CS485 project:
# - Sets up .github, .vscode, config, homeworks, final_project, scripts, templates, etc.
# - Creates missing directories/files based on templates.
#
# Run with:
#   ./scaffold_all.sh

# Define directories
DIRS=(
  ".github"
  ".vscode"
  "config"
  "homeworks/hw1"    # Assuming HW1 is complete; for other HW's, use hw2, hw3, etc.
  "scripts"
  "templates"
  "final_project"
  "final_project/src"
  "final_project/include"
  "final_project/data"
  "final_project/docs"
  "final_project/ai_chat"
)

echo "Setting up repository structure..."

for d in "${DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "Creating directory: $d"
        mkdir -p "$d"
    else
        echo "Directory exists: $d"
    fi
done

echo "Setting up VS Code configuration..."
# Create default VS Code settings if not present
if [ ! -f ".vscode/c_cpp_properties.json" ]; then
cat > .vscode/c_cpp_properties.json << 'EOF'
{
    "configurations": [
        {
            "name": "WSL",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/lib/x86_64-linux-gnu/openmpi/include",
                "/usr/local/cuda-12.8/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc-12",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "gcc-x64"
        }
    ],
    "version": 4
}
EOF
fi

if [ ! -f ".vscode/settings.json" ]; then
cat > .vscode/settings.json << 'EOF'
{
    "cmake.configureOnOpen": false,
    "[makefile]": {
        "editor.insertSpaces": false
    }
}
EOF
fi

echo "Setting up cluster config template..."
if [ ! -f "config/cluster.conf.template" ]; then
cat > config/cluster.conf.template << 'EOF'
# Example cluster configuration
# Format: <hostname> <IP address> <username>
node1 10.0.0.1 user1
node2 10.0.0.2 user2
EOF
fi

echo "Setting up homework HW1..."
# Create default files for HW1 if missing
HW1_DIR="homeworks/hw1"
if [ ! -d "$HW1_DIR/src" ]; then
    mkdir -p "$HW1_DIR/src"
fi
if [ ! -f "$HW1_DIR/src/template.c" ]; then
cat > "$HW1_DIR/src/template.c" << 'EOF'
/*
 * HW1: MPI Point-to-Point Matrix Multiplication
 * This is the template code for Homework 1.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int n = 128;  // Default problem size
    MPI_Init(&argc, &argv);
    // Insert matrix multiplication logic here (using MPI_Send and MPI_Recv)
    // ...
    MPI_Finalize();
    return 0;
}
EOF
fi
if [ ! -f "$HW1_DIR/Makefile" ]; then
cat > "$HW1_DIR/Makefile" << 'EOF'
# Makefile for HW1 - builds an executable named 'template'
CC = mpicc
CFLAGS = -O2 -std=c11

template: src/template.c
	$(CC) $(CFLAGS) -o template src/template.c

.PHONY: clean
clean:
	rm -f template
EOF
fi
if [ ! -f "$HW1_DIR/summary.md" ]; then
cat > "$HW1_DIR/summary.md" << 'EOF'
# Homework 1 Summary

- **Topic:** MPI Point-to-Point Matrix Multiplication
- **Approach:** Matrix A is partitioned among processes; matrix B is broadcast.
- **Result:** Correct execution for valid combinations; tests confirm expected timeouts for oversized problems.
- **Lessons Learned:** TABs in Makefiles, MPI send/recv usage, handling n % np conditions.
EOF
fi

echo "Setting up templates..."
# Check and populate templates if not present
if [ ! -f "templates/CMakeLists.txt.template" ]; then
cat > templates/CMakeLists.txt.template << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(hwX_project C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
add_executable(template src/template.c)  # or change to template.cu for CUDA homework

target_link_libraries(template ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
EOF
fi
if [ ! -f "templates/template.c.template" ]; then
cat > templates/template.c.template << 'EOF'
/*
 * Template C file for homework submissions.
 * Author: <Your Name>
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    // Your code here.
    MPI_Finalize();
    return 0;
}
EOF
fi
if [ ! -f "templates/template.cu.template" ]; then
cat > templates/template.cu.template << 'EOF'
/*
 * Template CUDA file for homework submissions.
 * Author: <Your Name>
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void kernel() {
    // Your CUDA kernel code here.
}

int main(int argc, char *argv[]) {
    // Launch kernel with dummy parameters
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF
fi

echo "Setting up final project structure (AlexNet MPI+CUDA)..."
FP_DIR="final_project"
# Create a basic CMakeLists.txt for the final project
if [ ! -f "$FP_DIR/CMakeLists.txt" ]; then
cat > "$FP_DIR/CMakeLists.txt" << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(AlexNet_MPI_CUDA C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} include)

# Add the hybrid MPI+CUDA AlexNet executable
add_executable(alexnet_hybrid src/alexnet_hybrid.cu)
target_link_libraries(alexnet_hybrid ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
EOF
fi

# Create a simple Makefile for the final project submission
if [ ! -f "$FP_DIR/Makefile" ]; then
cat > "$FP_DIR/Makefile" << 'EOF'
# Makefile for final project (AlexNet MPI+CUDA)
# This Makefile should build an executable named 'template'
CC = mpicxx
NVCC = nvcc
CFLAGS = -O2 -std=c++17

SRC = $(wildcard src/*.cpp) $(wildcard src/*.cu)
OBJ = $(SRC:.cpp=.o)

template: $(SRC)
	$(NVCC) $(CFLAGS) -o template $(SRC) $(shell pkg-config --cflags --libs cuda) $(shell mpicc --showme:libs)

.PHONY: clean
clean:
	rm -f template
EOF
fi

# Create placeholder source files for the final project if missing
if [ ! -f "$FP_DIR/src/alexnet_hybrid.cu" ]; then
cat > "$FP_DIR/src/alexnet_hybrid.cu" << 'EOF'
/*
 * Final Project: AlexNet MPI+CUDA Implementation
 * This file contains the hybrid MPI+CUDA code for distributed inference/training of AlexNet.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

// (Insert AlexNet layer definitions, CUDA kernels, and MPI communication logic here)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    // For example, read input, broadcast model parameters, perform local GPU computation, and synchronize via MPI_Allreduce.
    // Note: Replace the following with actual AlexNet implementation code.
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Hello from rank %d\n", rank);

    // Dummy CUDA kernel call
    cudaSetDevice(0);
    // ... launch kernels, etc.
    MPI_Finalize();
    return 0;
}
EOF
fi

# Create a sample .curorfile for AI Chat Assistant (in final_project/ai_chat)
if [ ! -f "$FP_DIR/ai_chat/.curorfile" ]; then
cat > "$FP_DIR/ai_chat/.curorfile" << 'EOF'
{
    "system_prompt": "You are an expert AI assistant specialized in high performance computing, distributed GPU programming, MPI, and CUDA. Provide clear, step-by-step help for the AlexNet implementation project. Answer questions with code examples, best practices, and troubleshooting tips."
}
EOF
fi

echo "Repository scaffolding complete."
echo "Please review the generated files and directories, then commit changes to Git."
echo "You are now ready to move ahead with the final project - full steam ahead on implementing AlexNet!"
