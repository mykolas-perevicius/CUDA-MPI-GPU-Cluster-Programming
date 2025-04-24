#!/usr/bin/env bash

# Test script specifically for V4 MPI+CUDA

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Pipestatus: exit code of the last command to exit non-zero is returned.
set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_v4_test" # Dedicated log directory for this test
MAKE_CMD="make"
MPI_RUN_CMD="mpirun"
MPI_ARGS="--oversubscribe" # Use for local testing, remove/adjust for cluster SLURM/PBS
NP_VALUES=(1 2 4)          # Process counts to test
TARGET_EXEC="template"

# --- Colors for Output ---
COL_RESET=$(tput sgr0 || echo "")
COL_GREEN=$(tput setaf 2 || echo "")
COL_RED=$(tput setaf 1 || echo "")
COL_YELLOW=$(tput setaf 3 || echo "")
COL_BLUE=$(tput setaf 4 || echo "")

# --- Helper Functions ---
log_info() {
    echo "${COL_BLUE}INFO: $*${COL_RESET}"
}

log_pass() {
    echo "${COL_GREEN}PASS: $*${COL_RESET}"
}

log_fail() {
    echo "${COL_RED}FAIL: $*${COL_RESET}"
}

log_warn() {
    echo "${COL_YELLOW}WARN: $*${COL_RESET}"
}

# --- Main Script ---
cd "$SCRIPT_DIR" # Ensure we are in the v4_mpi_cuda directory
mkdir -p "$LOG_DIR"
log_info "Created log directory: $LOG_DIR"

# 1. Build Step
log_info "Attempting to build V4 ($MAKE_CMD)..."
BUILD_LOG="${LOG_DIR}/build.log"
touch "$BUILD_LOG" && > "$BUILD_LOG" # Clear log

if ! "$MAKE_CMD" clean >> "$BUILD_LOG" 2>&1; then
    log_warn "Make clean potentially failed (check $BUILD_LOG), continuing build..."
fi

if ! "$MAKE_CMD" >> "$BUILD_LOG" 2>&1; then
    log_fail "Build failed. See $BUILD_LOG for details."
    # Display last few lines of build log for quick diagnosis
    echo "--- Last 15 lines of build log ($BUILD_LOG): ---"
    tail -n 15 "$BUILD_LOG"
    echo "----------------------------------------------"
    exit 1
else
    log_pass "Build successful."
fi

# Check if target executable exists
if [ ! -f "$TARGET_EXEC" ]; then
    log_fail "Target executable '$TARGET_EXEC' not found after successful build report. Check Makefile."
    exit 1
fi

# 2. Run Tests
declare -a failed_runs
declare -a warning_runs
declare -A run_times # Associative array to store times

log_info "Starting runtime tests for np=${NP_VALUES[*]}..."

for np in "${NP_VALUES[@]}"; do
    echo # Blank line for spacing
    log_info "Running V4 with np=$np..."
    LOG_FILE="${LOG_DIR}/v4_np${np}.log"
    touch "$LOG_FILE" && > "$LOG_FILE" # Clear log

    # Construct the command
    FULL_CMD="$MPI_RUN_CMD $MPI_ARGS -np $np ./${TARGET_EXEC}"
    echo "-> Executing: $FULL_CMD"

    # Execute and capture exit code
    run_exit_code=0
    "$MPI_RUN_CMD" $MPI_ARGS -np "$np" ./"${TARGET_EXEC}" >> "$LOG_FILE" 2>&1 || run_exit_code=$?

    if [[ $run_exit_code -eq 0 ]]; then
        log_pass "np=$np completed successfully."
        # Try parsing time (optional, for summary)
        time_v4=$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$LOG_FILE" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "N/A")
        run_times[$np]=$time_v4
        # Check for internal warnings printed by the C++ code
        if grep -q -i "WARNING:" "$LOG_FILE"; then
             log_warn "np=$np finished, but C++ code printed WARNINGS (check $LOG_FILE)."
             warning_runs+=("np=$np (Internal Warning)")
        fi
    else
        log_fail "np=$np failed with exit code $run_exit_code."
        failed_runs+=("np=$np (Exit Code: $run_exit_code)")
        run_times[$np]="FAIL"

        # Provide detailed error context
        echo "--- Potential errors from log (${LOG_FILE}): ---"
        # Look for common error patterns
        grep -iE 'error|fail|abort|fault|warning|signal' "$LOG_FILE" | head -n 15 || echo "[No specific error keywords found by grep]"
        echo "--- Last 10 lines of log (${LOG_FILE}): ---"
        tail -n 10 "$LOG_FILE"
        echo "--------------------------------------------------"
        # Optionally add specific checks for CUDA/MPI errors if needed
        # if grep -q "CUDA error" "$LOG_FILE"; then ... fi
    fi
done

# 3. Final Summary
echo # Blank line
log_info "============== V4 Test Summary =============="
log_info "Build: Successful"
log_info "Logs Location: $LOG_DIR"
echo "--- Runtime Results ---"
for np in "${NP_VALUES[@]}"; do
    time_str="${run_times[$np]:-N/A}"
    status_msg=""
    final_col=$COL_RESET
    found_fail=0
    found_warn=0

    for failed in "${failed_runs[@]}"; do
        if [[ "$failed" == "np=$np"* ]]; then
            status_msg="${COL_RED}[ FAILED ]${COL_RESET}"
            final_col=$COL_RED
            found_fail=1
            break
        fi
    done

    if [[ $found_fail -eq 0 ]]; then
         for warned in "${warning_runs[@]}"; do
              if [[ "$warned" == "np=$np"* ]]; then
                  status_msg="${COL_YELLOW}[ WARNING ]${COL_RESET}"
                  final_col=$COL_YELLOW
                  found_warn=1
                  break
              fi
         done
    fi

     if [[ $found_fail -eq 0 && $found_warn -eq 0 ]]; then
        status_msg="${COL_GREEN}[ PASSED ]${COL_RESET}"
        final_col=$COL_GREEN
    fi

    printf "np=%-2d: %s ${final_col}(Time: %s)${COL_RESET}\n" "$np" "$status_msg" "$time_str"
done
echo "============================================="

# Exit with non-zero code if any runs failed
if [ ${#failed_runs[@]} -gt 0 ]; then
    exit 1
fi

exit 0