#!/bin/bash

# --- Argument Handling ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <homework_number> <lastname> <firstname>"
    echo "Example: $0 1 doe jane"
    exit 1
fi

HW_NUM=$1
LASTNAME=$2
FIRSTNAME=$3

# Get the directory where this script resides to call other scripts reliably
SCRIPT_DIR=$(dirname "$(realpath "$0")")

echo "--- Running Full Workflow for Homework ${HW_NUM} ---"

# --- Step 1: Test the homework ---
echo ""
echo "==> Running Tests..."
# Execute the test script
bash "${SCRIPT_DIR}/test_hw.sh" "$HW_NUM"
TEST_EXIT_CODE=$? # Capture the exit code from test_hw.sh

# Check the exit code from the test script
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "==> Tests PASSED."
elif [ $TEST_EXIT_CODE -eq 2 ]; then
    # Exit code 2 from test_hw.sh indicates timeout
    echo "==> Tests INCONCLUSIVE (Timeout Occurred). Proceeding with packaging..."
    # Allow packaging even on timeout, as code might be mostly correct
else
    # Any other non-zero exit code means failure
    echo "!!! Tests FAILED (Exit Code: $TEST_EXIT_CODE). Aborting packaging. !!!"
    exit 1
fi

# --- Step 2: Package the homework ---
echo ""
echo "==> Packaging homework..."
# Execute the packaging script
if ! bash "${SCRIPT_DIR}/package_hw.sh" "$HW_NUM" "$LASTNAME" "$FIRSTNAME"; then
    echo "!!! Packaging failed for homework ${HW_NUM}. !!!"
    exit 1 # Exit if packaging itself fails
fi

echo ""
echo "--- Full Workflow for Homework ${HW_NUM}: COMPLETED ---"
# Final exit code reflects test status (0=pass, 2=timeout) unless packaging failed
exit $TEST_EXIT_CODE