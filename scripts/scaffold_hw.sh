#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <homework_number>"
    exit 1
fi

HW_NUM=$1
HW_DIR="homeworks/hw${HW_NUM}"
SRC_DIR="$HW_DIR/src"
TEMPLATES_DIR="templates"

echo "Scaffolding homework ${HW_NUM} in ${HW_DIR}..."

# Create directories
mkdir -p "$SRC_DIR" || { echo "Failed to create directory $SRC_DIR"; exit 1; }
# Build directory will be created by test script if needed, or manually

# Create summary file
touch "$HW_DIR/summary.md"
echo "# Summary for Homework ${HW_NUM}" > "$HW_DIR/summary.md"
echo "" >> "$HW_DIR/summary.md"
echo "## Problem Description" >> "$HW_DIR/summary.md"
echo "" >> "$HW_DIR/summary.md"
echo "## Key Concepts" >> "$HW_DIR/summary.md"
echo "" >> "$HW_DIR/summary.md"
echo "## Potential Issues & Debugging" >> "$HW_DIR/summary.md"
echo "" >> "$HW_DIR/summary.md"
echo "## Performance Notes" >> "$HW_DIR/summary.md"
echo "" >> "$HW_DIR/summary.md"
echo "## Potential Exam Questions" >> "$HW_DIR/summary.md"

# Decide on build system and source template based on homework number
# Homeworks 1-3: MPI Only (Makefile, .c)
# Homeworks 4-9: MPI + CUDA (CMake recommended for dev, Makefile for submission, .cu)
if [ "$HW_NUM" -ge 1 ] && [ "$HW_NUM" -le 3 ]; then
    echo "Setting up for MPI Only (Makefile, .c)"
    cp "$TEMPLATES_DIR/Makefile.template" "$HW_DIR/Makefile" || echo "Warning: Failed to copy Makefile template."
    cp "$TEMPLATES_DIR/template.c.template" "$SRC_DIR/template.c" || echo "Warning: Failed to copy C template."
elif [ "$HW_NUM" -ge 4 ]; then
    echo "Setting up for MPI+CUDA (CMakeLists.txt, Makefile for submission, .cu)"
    cp "$TEMPLATES_DIR/CMakeLists.txt.template" "$HW_DIR/CMakeLists.txt" || echo "Warning: Failed to copy CMakeLists template."
    # IMPORTANT: Provide the simple Makefile needed for submission too!
    cp "$TEMPLATES_DIR/Makefile.template" "$HW_DIR/Makefile" || echo "Warning: Failed to copy submission Makefile template."
    # You might need to adjust the submission Makefile for CUDA linking manually later.
    cp "$TEMPLATES_DIR/template.cu.template" "$SRC_DIR/template.cu" || echo "Warning: Failed to copy CU template."
    # Rename if using C++ (.cpp)
    # mv "$SRC_DIR/template.cu" "$SRC_DIR/template.cpp"
else
    echo "Warning: Homework number ${HW_NUM} out of expected range (1-9)."
fi

echo "Scaffolding complete for homework ${HW_NUM}."
echo "Remember to edit:"
echo "  - ${SRC_DIR}/template.c (or .cu)"
echo "  - ${HW_DIR}/Makefile (especially for submission if using CMake for dev)"
echo "  - ${HW_DIR}/CMakeLists.txt (if applicable)"
echo "  - ${HW_DIR}/summary.md"