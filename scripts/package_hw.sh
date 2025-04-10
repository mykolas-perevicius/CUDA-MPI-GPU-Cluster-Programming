#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <homework_number> <lastname> <firstname>"
    exit 1
fi

HW_NUM=$1
LASTNAME=$2
FIRSTNAME=$3

HW_DIR="homeworks/hw${HW_NUM}"
SRC_DIR="$HW_DIR/src"
SUBMISSION_NAME="hw${HW_NUM}-${LASTNAME}-${FIRSTNAME}"
ARCHIVE_NAME="${SUBMISSION_NAME}.tgz"

REQUIRED_MAKEFILE="$HW_DIR/Makefile" # The makefile NEEDED for submission
# Determine required source file (.c or .cu or .cpp)
REQUIRED_SRC=""
if [ -f "$SRC_DIR/template.cu" ]; then
    REQUIRED_SRC="$SRC_DIR/template.cu"
elif [ -f "$SRC_DIR/template.c" ]; then
     REQUIRED_SRC="$SRC_DIR/template.c"
elif [ -f "$SRC_DIR/template.cpp" ]; then
     REQUIRED_SRC="$SRC_DIR/template.cpp"
else
     echo "Error: No template.c, template.cu, or template.cpp found in $SRC_DIR"
     exit 1
fi

echo "--- Packaging Homework ${HW_NUM} for ${FIRSTNAME} ${LASTNAME} ---"

# --- Validations ---
if [ ! -d "$HW_DIR" ]; then
    echo "Error: Homework directory '$HW_DIR' not found."
    exit 1
fi
 if [ ! -f "$REQUIRED_SRC" ]; then
    echo "Error: Required source file '$REQUIRED_SRC' not found."
    exit 1
fi
 if [ ! -f "$REQUIRED_MAKEFILE" ]; then
    echo "Error: Required submission Makefile '$REQUIRED_MAKEFILE' not found."
    echo "Ensure a simple Makefile exists in $HW_DIR, even if you used CMake for development."
    exit 1
fi

# --- Create temporary directory for packaging ---
echo "Creating temporary packaging directory: $SUBMISSION_NAME"
rm -rf "$SUBMISSION_NAME" "$ARCHIVE_NAME" # Clean previous attempts
mkdir "$SUBMISSION_NAME" || { echo "Failed to create temp dir"; exit 1; }

# --- Copy required files ---
echo "Copying required files..."
cp "$REQUIRED_SRC" "$SUBMISSION_NAME/" || { echo "Failed to copy source"; rm -r "$SUBMISSION_NAME"; exit 1; }
cp "$REQUIRED_MAKEFILE" "$SUBMISSION_NAME/" || { echo "Failed to copy Makefile"; rm -r "$SUBMISSION_NAME"; exit 1; }

# --- Create Archive ---
echo "Creating archive: $ARCHIVE_NAME"
tar czf "$ARCHIVE_NAME" "$SUBMISSION_NAME" || { echo "Failed to create tar archive"; rm -r "$SUBMISSION_NAME"; exit 1; }

# --- Cleanup ---
echo "Cleaning up temporary directory..."
rm -r "$SUBMISSION_NAME"

# --- Final Check ---
if [ -f "$ARCHIVE_NAME" ]; then
    echo "--- Homework ${HW_NUM} Packaging: SUCCESS ---"
    echo "Archive created: $ARCHIVE_NAME"
    exit 0
else
    echo "--- Homework ${HW_NUM} Packaging: FAILED ---"
    echo "Archive $ARCHIVE_NAME not found."
    exit 1
fi