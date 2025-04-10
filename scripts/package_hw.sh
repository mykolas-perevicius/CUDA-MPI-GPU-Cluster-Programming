#!/bin/bash

# --- Argument Handling ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <homework_number> <lastname> <firstname>"
    echo "Example: $0 1 doe jane"
    exit 1
fi

HW_NUM=$1
LASTNAME=$(echo "$2" | tr '[:upper:]' '[:lower:]') # Ensure lowercase lastname
FIRSTNAME=$(echo "$3" | tr '[:upper:]' '[:lower:]') # Ensure lowercase firstname

# --- Configuration ---
HW_DIR="homeworks/hw${HW_NUM}"
SRC_DIR="$HW_DIR/src"
# Construct submission directory and archive names per course spec
SUBMISSION_DIR_NAME="hw${HW_NUM}-${LASTNAME}-${FIRSTNAME}"
ARCHIVE_NAME="${SUBMISSION_DIR_NAME}.tgz"
# Define the Makefile required for submission (located directly in HW_DIR)
SUBMISSION_MAKEFILE="$HW_DIR/Makefile"

# --- Determine required source file (.c or .cu or .cpp) ---
# Prioritize .cu, then .c, then .cpp
SUBMISSION_SRC=""
if [ -f "$SRC_DIR/template.cu" ]; then
    SUBMISSION_SRC="$SRC_DIR/template.cu"
elif [ -f "$SRC_DIR/template.c" ]; then
     SUBMISSION_SRC="$SRC_DIR/template.c"
elif [ -f "$SRC_DIR/template.cpp" ]; then
     SUBMISSION_SRC="$SRC_DIR/template.cpp"
else
     echo "Error: No template source file (template.c, template.cu, or template.cpp) found in $SRC_DIR"
     exit 1
fi
SUBMISSION_SRC_BASENAME=$(basename "$SUBMISSION_SRC") # Get just the filename (e.g., template.c)

echo "--- Packaging Homework ${HW_NUM} for ${FIRSTNAME} ${LASTNAME} ---"
echo "Source file: $SUBMISSION_SRC"
echo "Makefile: $SUBMISSION_MAKEFILE"
echo "Output Archive: $ARCHIVE_NAME"

# --- Validations ---
if [ ! -d "$HW_DIR" ]; then
    echo "Error: Homework directory '$HW_DIR' not found."
    exit 1
fi
 if [ ! -f "$SUBMISSION_SRC" ]; then
    # This check is somewhat redundant due to earlier logic, but good safety
    echo "Error: Required source file '$SUBMISSION_SRC' not found."
    exit 1
fi
 if [ ! -f "$SUBMISSION_MAKEFILE" ]; then
    echo "Error: Required submission Makefile '$SUBMISSION_MAKEFILE' not found."
    echo "Ensure this Makefile exists and is correct for grading."
    exit 1
fi

# --- Create temporary directory for packaging ---
echo "Creating temporary packaging directory: $SUBMISSION_DIR_NAME"
# Clean previous attempts in the root directory
rm -rf "$SUBMISSION_DIR_NAME" "$ARCHIVE_NAME"
mkdir "$SUBMISSION_DIR_NAME" || { echo "Failed to create temp dir '$SUBMISSION_DIR_NAME'"; exit 1; }

# --- Copy required files into the temporary directory ---
echo "Copying required files into $SUBMISSION_DIR_NAME ..."
# Copy source file, keeping its original name
cp "$SUBMISSION_SRC" "$SUBMISSION_DIR_NAME/" || { echo "Failed to copy source '$SUBMISSION_SRC'"; rm -rf "$SUBMISSION_DIR_NAME"; exit 1; }
# Copy the submission Makefile, keeping its name
cp "$SUBMISSION_MAKEFILE" "$SUBMISSION_DIR_NAME/" || { echo "Failed to copy Makefile '$SUBMISSION_MAKEFILE'"; rm -rf "$SUBMISSION_DIR_NAME"; exit 1; }

# --- Verify files exist in the staging directory ---
if [ ! -f "$SUBMISSION_DIR_NAME/$SUBMISSION_SRC_BASENAME" ] || [ ! -f "$SUBMISSION_DIR_NAME/Makefile" ]; then
    echo "Error: Failed to verify files copied into staging directory '$SUBMISSION_DIR_NAME'."
    rm -rf "$SUBMISSION_DIR_NAME"
    exit 1
fi

# --- Create Archive ---
echo "Creating archive: $ARCHIVE_NAME from directory $SUBMISSION_DIR_NAME"
# Create the tar.gz archive from the temporary directory
tar czf "$ARCHIVE_NAME" "$SUBMISSION_DIR_NAME" || { echo "Failed to create tar archive"; rm -rf "$SUBMISSION_DIR_NAME"; exit 1; }

# --- Cleanup ---
echo "Cleaning up temporary directory '$SUBMISSION_DIR_NAME'..."
rm -rf "$SUBMISSION_DIR_NAME"

# --- Final Check ---
if [ -f "$ARCHIVE_NAME" ]; then
    echo "--- Homework ${HW_NUM} Packaging: SUCCESS ---"
    echo "Archive created: $ARCHIVE_NAME"
    exit 0
else
    echo "--- Homework ${HW_NUM} Packaging: FAILED ---"
    echo "Archive $ARCHIVE_NAME not found after attempt."
    exit 1
fi