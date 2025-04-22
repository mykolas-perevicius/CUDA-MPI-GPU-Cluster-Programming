#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root directory:
# ~/CS485/CUDA-MPI-GPU-Cluster-Programming$ bash collect_project.sh

ROOT_DIR="$(pwd)"
OUTPUT_FILE="project.txt"
PROJECT_BASE="final_project"

# List of files to collect relative to PROJECT_BASE
FILES_TO_COLLECT=(
    # V1 Serial - Main focus for current errors
    "v1_serial/Makefile"
    "v1_serial/include/alexnet.hpp"
    "v1_serial/include/layers.hpp"
    "v1_serial/src/main.cpp"
    "v1_serial/src/alexnet_serial.cpp"
    "v1_serial/src/layers_serial.cpp"

    # V2 MPI - Just include main for context if needed
    "v2_mpi_only/2.2_scatter_halo/include/alexnet.hpp"
    "v2_mpi_only/2.2_scatter_halo/include/layers.hpp"
    "v2_mpi_only/2.2_scatter_halo/src/main.cpp"
    "v2_mpi_only/2.2_scatter_halo/src/alexnet_mpi.cpp"
    "v2_mpi_only/2.2_scatter_halo/src/layers_mpi.cpp"


    # V3 CUDA - For the IDE hint
    "v3_cuda_only/Makefile"
    "v3_cuda_only/include/alexnet.hpp"
    "v3_cuda_only/include/layers.hpp"
    "v3_cuda_only/src/main_cuda.cpp"
    "v3_cuda_only/src/alexnet_cuda.cu"
    "v3_cuda_only/src/layers_cuda.cu"
)

# Clear the output file
> "$OUTPUT_FILE"

echo "Collecting project files into $OUTPUT_FILE..."

# Loop through the files and append them to the output file
for relative_path in "${FILES_TO_COLLECT[@]}"; do
    full_path="$ROOT_DIR/$PROJECT_BASE/$relative_path"
    echo "Appending: $PROJECT_BASE/$relative_path"
    printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
    printf "\n\n=== FILE: %s ===\n\n" "$PROJECT_BASE/$relative_path" >> "$OUTPUT_FILE"

    if [[ -f "$full_path" ]]; then
        cat "$full_path" >> "$OUTPUT_FILE"
    else
        echo "*** ERROR: File not found! ***" >> "$OUTPUT_FILE"
        echo "Warning: Could not find $full_path" >&2 # Also print warning to terminal
    fi
    printf "\n\n" >> "$OUTPUT_FILE"
done

printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Done. Please copy the contents of $OUTPUT_FILE"

exit 0