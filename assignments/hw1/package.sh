#!/bin/bash
# package.sh: Verifies naming and packages Homework 1 for submission.
EXPECTED_DIR="hw1-doe-jane"
CURRENT_DIR=\$(basename "\$(pwd)")

if [ "\$CURRENT_DIR" != "\$EXPECTED_DIR" ]; then
    echo "Error: The current directory must be named \$EXPECTED_DIR"
    exit 1
fi

if [ ! -f "template" ]; then
    echo "Error: Executable 'template' not found. Build the project first."
    exit 1
fi

tar czf "\$EXPECTED_DIR.tgz" .
echo "Package created: \$EXPECTED_DIR.tgz"
