#!/bin/bash
# scripts/run_final_project_extended.sh
#
# This script automates:
# 1. Building the final project executable ("template") and the conv_test unit from final_project/.
# 2. Running conv_test to verify the convolution kernel functionality.
# 3. Running the final project executable with different MPI process counts and synthetic
#    data multipliers, each run wrapped in a 30-second timeout.
# 4. Collecting output logs and producing a concise summary.
#
# Process counts tested: 1, 2, and 4.
# Multipliers tested: 1, 2, and 4.
#
# Logs for MPI runs are stored in final_project/logs_extended/ and for conv_test in final_project/logs_extended/.
#
# Timeout duration:
TIMEOUT_DURATION=30

# Build the final project executable and conv_test.
echo "Building final_project and conv_test..."
cd ../final_project || { echo "final_project directory not found!"; exit 1; }
make clean
if ! make; then
    echo "Build failed. Exiting."
    exit 1
fi

# Build the conv_test target explicitly.
if ! make conv_test; then
    echo "conv_test build failed. Exiting."
    exit 1
fi
cd ..  # Return to repository root

# Define the executables.
EXEC="./final_project/template"
CONV_TEST_EXEC="./final_project/conv_test"

if [ ! -f "$EXEC" ]; then
    echo "Executable $EXEC not found after build. Exiting."
    exit 1
fi

if [ ! -f "$CONV_TEST_EXEC" ]; then
    echo "Executable $CONV_TEST_EXEC not found after build. Exiting."
    exit 1
fi

# Create log directory for extended tests.
LOG_DIR="final_project/logs_extended"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Running conv_test..."
CONV_LOGFILE="${LOG_DIR}/conv_test.log"
if timeout $TIMEOUT_DURATION "$CONV_TEST_EXEC" | tee "$CONV_LOGFILE"; then
    echo "conv_test completed successfully."
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "conv_test timed out after ${TIMEOUT_DURATION}s." | tee -a "$CONV_LOGFILE"
    else
        echo "conv_test failed with exit code $EXIT_CODE." | tee -a "$CONV_LOGFILE"
    fi
fi

echo "=============================================="
echo "Starting extended tests for final_project (MPI runs)..."

# Define MPI process counts and synthetic data multipliers.
NP_LIST=(1 2 4)
MULTIPLIER_LIST=(1 2 4)

for multiplier in "${MULTIPLIER_LIST[@]}"; do
    for np in "${NP_LIST[@]}"; do
        echo "----------------------------------------------"
        echo "Running with -np $np and multiplier $multiplier ..."
        LOGFILE="${LOG_DIR}/final_project_np${np}_mult${multiplier}.log"
        if timeout $TIMEOUT_DURATION mpirun --oversubscribe -np "$np" "$EXEC" "$multiplier" | tee "$LOGFILE"; then
            echo "Run with -np $np, multiplier $multiplier completed successfully."
        else
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                echo "Run with -np $np, multiplier $multiplier timed out after ${TIMEOUT_DURATION}s." | tee -a "$LOGFILE"
            else
                echo "Run with -np $np, multiplier $multiplier failed with exit code $EXIT_CODE." | tee -a "$LOGFILE"
            fi
        fi
    done
done

echo "----------------------------------------------"
echo "Extended tests completed. Logs are available in ${LOG_DIR}/"

echo "=============================================="
echo "Final Summary:"
echo "Convolution Test Summary (conv_test.log):"
grep -E "(Convolution Test Output|error|timed out)" "${LOG_DIR}/conv_test.log" || echo "No summary info in conv_test.log"
echo "----------------------------------------------"
for log in "$LOG_DIR"/*.log; do
    # Skip conv_test.log because we printed it already.
    if [[ $(basename "$log") == "conv_test.log" ]]; then
        continue
    fi
    echo "Summary for $(basename "$log"):"
    grep -E "(forward pass time \(MPI_Wtime\)|Kernel execution time:)" "$log"
    echo "----------------------------------------------"
done

# Aggregate summary: Compute the average forward pass time per log file and print all results on one line.
echo "----------------------------------------------"
echo "Aggregate Final Summary:"
aggregate_summary=""
for log in "${LOG_DIR}"/*.log; do
    # Extract "forward pass time" lines and average the last field (which is the time in seconds).
    avg_time=$(grep -E "forward pass time \(MPI_Wtime\):" "$log" | \
               awk '{sum += $NF; count++} END {if(count>0) printf "%.6f", sum/count; else print "N/A"}')
    file=$(basename "$log")
    aggregate_summary="${aggregate_summary} ${file}: avgForward=${avg_time}s |"
done
# Print the final single-line summary.
echo "$aggregate_summary"
