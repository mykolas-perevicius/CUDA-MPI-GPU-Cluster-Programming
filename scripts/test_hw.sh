#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <homework_number>"
    exit 1
fi

HW_NUM=$1
HW_DIR="homeworks/hw${HW_NUM}"
BUILD_DIR="$HW_DIR/build" # Build directory for CMake
EXE_NAME="template"
SRC_DIR="$HW_DIR/src" # Source location

# --- Validation ---
if [ ! -d "$HW_DIR" ]; then
    echo "Error: Homework directory '$HW_DIR' not found."
    echo "Did you run 'scripts/scaffold_hw.sh $HW_NUM'?"
    exit 1
fi

echo "--- Testing Homework ${HW_NUM} ---"
cd "$HW_DIR" || exit 1 # Enter homework directory

# --- Build Step ---
echo "Building homework ${HW_NUM}..."
BUILD_SUCCESS=0
# Prefer CMake if CMakeLists.txt exists, otherwise use Makefile
if [ -f "CMakeLists.txt" ]; then
    echo "Using CMake build system..."
    mkdir -p "$BUILD_DIR"
    if ! cmake -S . -B "$BUILD_DIR"; then
        echo "CMake configuration failed."
        cd ..
        exit 1
    fi
    if ! cmake --build "$BUILD_DIR"; then
         echo "CMake build failed."
         cd ..
         exit 1
    fi
    # Executable should be in build dir for CMake
    EXE_PATH="$BUILD_DIR/$EXE_NAME"
    BUILD_SUCCESS=1
elif [ -f "Makefile" ]; then
    echo "Using Makefile build system..."
    make clean > /dev/null 2>&1
    if ! make; then
         echo "Make build failed."
         cd ..
         exit 1
    fi
     # Executable should be in the current dir for simple Make
    EXE_PATH="./$EXE_NAME"
    BUILD_SUCCESS=1
else
    echo "Error: No Makefile or CMakeLists.txt found in $HW_DIR."
    cd ..
    exit 1
fi

if [ ! -x "$EXE_PATH" ]; then
     echo "Error: Executable '$EXE_PATH' not found or not executable after build."
     cd ..
     exit 1
fi
echo "Build successful: $EXE_PATH"

# --- Test Execution Step (Simulate Prof's Script) ---
echo "Running tests (simulating np=1-8, size=128-2048)..."
TEST_FAILED=0
# Define problem sizes and process counts based on course info
PROBLEM_SIZES=(128 256 512 1024 2048)
PROCESS_COUNTS=(1 2 3 4 5 6 7 8) # Adjust if needed

for np in "${PROCESS_COUNTS[@]}"; do
    for size in "${PROBLEM_SIZES[@]}"; do
        echo "Running: mpirun -np $np $EXE_PATH $size"
        # Use --oversubscribe if running more processes than cores locally
        mpirun --oversubscribe -np "$np" "$EXE_PATH" "$size"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "!!! Test Failed: np=$np, size=$size exited with code $EXIT_CODE !!!"
            TEST_FAILED=1
            # Decide if you want to stop on first failure or run all tests
            # break 2 # Break outer loops
        else
             echo "    Success: np=$np, size=$size"
        fi
    done
done

# --- Report Result ---
cd .. # Go back to root directory
if [ $TEST_FAILED -eq 0 ]; then
    echo "--- Homework ${HW_NUM} Testing: PASSED ---"
    exit 0
else
    echo "--- Homework ${HW_NUM} Testing: FAILED ---"
    exit 1
fi