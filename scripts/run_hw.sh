#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <homework_number> <lastname> <firstname>"
    exit 1
fi

HW_NUM=$1
LASTNAME=$2
FIRSTNAME=$3

SCRIPT_DIR=$(dirname "$0") # Get directory where this script resides

echo "--- Running Full Workflow for Homework ${HW_NUM} ---"

# Step 1: Test the homework
echo "Running tests..."
if ! bash "${SCRIPT_DIR}/test_hw.sh" "$HW_NUM"; then
    echo "!!! Testing failed for homework ${HW_NUM}. Aborting packaging. !!!"
    exit 1
fi

echo ""
echo "Tests passed."

# Step 2: Package the homework
echo "Packaging homework..."
if ! bash "${SCRIPT_DIR}/package_hw.sh" "$HW_NUM" "$LASTNAME" "$FIRSTNAME"; then
    echo "!!! Packaging failed for homework ${HW_NUM}. !!!"
    exit 1
fi

echo ""
echo "--- Full Workflow for Homework ${HW_NUM}: COMPLETED SUCCESSFULLY ---"
exit 0