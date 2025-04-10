#!/bin/bash

# --- Configuration ---
# Timeout duration for each individual mpirun command
TIMEOUT_DURATION="30s"
# Default executable name expected by the course/Makefile
EXE_NAME="template"
# Define problem sizes and process counts based on course info and professor's test script
PROBLEM_SIZES=(128 256 512 1024 2048)
PROCESS_COUNTS=(1 2 3 4 5 6 7 8) # Runs np=1 through np=8

# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "Usage: $0 <homework_number>"
    echo "Example: $0 1"
    exit 1
fi
HW_NUM=$1
HW_DIR="homeworks/hw${HW_NUM}"
BUILD_DIR="build" # Build directory *inside* the specific homework folder

# --- Validation ---
if [ ! -d "$HW_DIR" ]; then
    echo "Error: Homework directory '$HW_DIR' not found."
    echo "Did you run 'scripts/scaffold_hw.sh $HW_NUM'?"
    exit 1
fi

echo "--- Testing Homework ${HW_NUM} ---"

# --- Directory Navigation ---
# Get the absolute path of the script's directory
SCRIPT_REAL_DIR=$(dirname "$(realpath "$0")")
# Construct the absolute path to the target homework directory
ABS_HW_DIR="${SCRIPT_REAL_DIR}/../${HW_DIR}"
# Change into the homework directory. Exit if it fails.
cd "$ABS_HW_DIR" || { echo "Error: Failed to cd into '$ABS_HW_DIR'"; exit 1; }
echo "Changed directory to: $(pwd)"

# --- Build Step ---
echo "Building homework ${HW_NUM}..."
BUILD_SUCCESS=0
EXE_PATH="" # Path to the final executable

# Determine build system based on homework number
# HW 1-3 use Makefile directly.
# HW 4+ use CMake for development builds (test_hw), but packaging uses Makefile.
if [ "$HW_NUM" -ge 1 ] && [ "$HW_NUM" -le 3 ]; then
     if [ -f "Makefile" ]; then
        echo "Using Makefile build system for HW ${HW_NUM}..."
        # Clean previous build artifacts if they exist in current dir
        make clean > /dev/null 2>&1
        # Attempt to build using make
        if ! make; then
             echo "Make build failed."
             cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
             exit 1
        fi
         # Executable should be in the current dir for simple Make
        EXE_PATH="./$EXE_NAME"
        BUILD_SUCCESS=1
    else
        echo "Error: Makefile not found in $(pwd) for HW ${HW_NUM}."
        cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
        exit 1
    fi
elif [ "$HW_NUM" -ge 4 ]; then
     if [ -f "CMakeLists.txt" ]; then
        echo "Using CMake build system for HW ${HW_NUM}..."
        # Use relative paths for build directory within this homework folder
        rm -rf "$BUILD_DIR" # Clean previous CMake build dir
        mkdir -p "$BUILD_DIR"
        # Configure using CMake (source is '.', build is './build')
        if ! cmake -S . -B "$BUILD_DIR"; then
            echo "CMake configuration failed."
            cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
            exit 1
        fi
        # Build using CMake build command
        if ! cmake --build "$BUILD_DIR"; then
             echo "CMake build failed."
             cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
             exit 1
        fi
        # Executable should be in the build dir for CMake
        EXE_PATH="$BUILD_DIR/$EXE_NAME"
        BUILD_SUCCESS=1
    else
        echo "Error: CMakeLists.txt not found in $(pwd) for HW ${HW_NUM}."
        cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
        exit 1
    fi
else
    echo "Error: Invalid Homework number ${HW_NUM}."
    cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
    exit 1
fi

# Verify executable exists and is executable
if [ ! -x "$EXE_PATH" ]; then
     echo "Error: Executable '$EXE_PATH' not found or not executable after build in $(pwd)."
     cd "${SCRIPT_REAL_DIR}/.." # Go back to root before exiting
     exit 1
fi
echo "Build successful: $EXE_PATH"

# --- Test Execution Step ---
echo "Running tests (simulating np=[${PROCESS_COUNTS[*]}], size=[${PROBLEM_SIZES[*]}]) with ${TIMEOUT_DURATION} timeout per run..."
TEST_FAILED=0    # Flag for tests that explicitly failed (non-zero exit code other than timeout)
TEST_TIMEOUT=0   # Flag for tests that were killed due to timeout
TEST_SKIPPED=0   # Counter for tests skipped due to invalid parameters (n % np != 0)

for np in "${PROCESS_COUNTS[@]}"; do
    for size in "${PROBLEM_SIZES[@]}"; do

        # Check if size is divisible by np. Skip if not, as the program is expected to abort.
        if (( size % np != 0 )); then
            echo "Skipping: np=$np, size=$size (size not divisible by np)"
            TEST_SKIPPED=$((TEST_SKIPPED + 1)) # Increment skipped count
            continue # Skip to the next iteration of the inner loop
        fi

        # Construct the command to run, including timeout and oversubscribe flag for local testing
        CMD="timeout ${TIMEOUT_DURATION} mpirun --oversubscribe -np ${np} ${EXE_PATH} ${size}"
        echo "Running: ${CMD}"

        # Execute mpirun command with timeout
        ${CMD} # Run the command directly
        EXIT_CODE=$? # Capture the exit code

        # Check the exit code from the 'timeout' command
        if [ $EXIT_CODE -eq 0 ]; then
            # Exit code 0 means the command completed successfully within the timeout
            echo "    Success: np=$np, size=$size"
        elif [ $EXIT_CODE -eq 124 ]; then
            # Exit code 124 is sent by 'timeout' when it kills the command
            echo "!!! Timeout ($TIMEOUT_DURATION): np=$np, size=$size !!!"
            TEST_TIMEOUT=1 # Mark that at least one timeout occurred
        else
            # Any other non-zero exit code means the underlying command (mpirun or template) failed
            echo "!!! Test Failed: np=$np, size=$size exited with code $EXIT_CODE !!!"
            TEST_FAILED=1 # Mark that at least one explicit failure occurred
            # Optional: Stop all tests on the first explicit failure
            # break 2 # Breaks both inner and outer loops
        fi
    done
done

# --- Report Result ---
# Go back to the original root directory before printing the final summary
cd "${SCRIPT_REAL_DIR}/.."
echo ""
echo "--- Test Summary for Homework ${HW_NUM} ---"

# Optionally report the number of skipped tests
if [ $TEST_SKIPPED -gt 0 ]; then
    echo "Tests Skipped (n % np != 0): $TEST_SKIPPED"
fi

OVERALL_STATUS="PASSED"
FINAL_EXIT_CODE=0 # Default exit code for overall success

if [ $TEST_FAILED -ne 0 ]; then
    # If any valid test explicitly failed, the overall result is FAILED
    echo "Status: FAILED (One or more valid tests failed execution)"
    OVERALL_STATUS="FAILED"
    FINAL_EXIT_CODE=1 # Use exit code 1 for explicit failure
elif [ $TEST_TIMEOUT -ne 0 ]; then
    # If no tests failed BUT at least one timed out, the result is INCONCLUSIVE
    echo "Status: INCONCLUSIVE (One or more valid tests timed out after ${TIMEOUT_DURATION})"
    OVERALL_STATUS="TIMEOUT"
    FINAL_EXIT_CODE=2 # Use exit code 2 for timeout/inconclusive
else
    # If no tests failed and no tests timed out, the overall result is PASSED
    echo "Status: PASSED (All valid tests completed successfully within time limit)"
fi
echo "--------------------------------------"

# Exit the script with the determined overall status code
exit $FINAL_EXIT_CODE