#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root directory:
# ~/CS485/CUDA-MPI-GPU-Cluster-Programming$ bash collect_project.sh

ROOT_DIR="$(pwd)"
OUTPUT_FILE="project.txt"
PROJECT_BASE="final_project"

# List of ALL potentially relevant files across versions
# The script will only include files that actually exist.
FILES_TO_COLLECT=(
    # V1 Serial - Baseline (Should exist)
    "v1_serial/Makefile"
    "v1_serial/include/alexnet.hpp"
    "v1_serial/include/layers.hpp"
    "v1_serial/src/main.cpp"
    "v1_serial/src/alexnet_serial.cpp"
    "v1_serial/src/layers_serial.cpp"

    # V2 MPI - Scatter+Halo Approach (V2.2) (Should exist)
    "v2_mpi_only/2.2_scatter_halo/Makefile"
    "v2_mpi_only/2.2_scatter_halo/include/alexnet.hpp"
    "v2_mpi_only/2.2_scatter_halo/include/layers.hpp"
    "v2_mpi_only/2.2_scatter_halo/src/main.cpp"
    "v2_mpi_only/2.2_scatter_halo/src/alexnet_mpi.cpp"
    "v2_mpi_only/2.2_scatter_halo/src/layers_mpi.cpp"

    # V3 CUDA - Single GPU Implementation (Should exist)
    "v3_cuda_only/Makefile"
    "v3_cuda_only/include/alexnet.hpp"
    "v3_cuda_only/include/layers.hpp"
    "v3_cuda_only/src/main_cuda.cpp" # Verify actual filename used
    "v3_cuda_only/src/alexnet_cuda.cu" # Verify actual filename used
    "v3_cuda_only/src/layers_cuda.cu" # Verify actual filename used

    # V4 MPI+CUDA - Hybrid (Include if exists)
    "v4_mpi_cuda/Makefile"
    "v4_mpi_cuda/include/alexnet.hpp"
    "v4_mpi_cuda/include/layers.hpp"
    "v4_mpi_cuda/src/main.cpp" # Anticipated main MPI driver
    "v4_mpi_cuda/src/alexnet_mpi_cuda.cu" # Anticipated hybrid orchestrator
    "v4_mpi_cuda/src/layers_cuda.cu" # Likely reused CUDA kernels

    # V5 CUDA-Aware MPI - Optimization (Include if exists)
    "v5_cuda_aware_mpi/Makefile"
    "v5_cuda_aware_mpi/include/alexnet.hpp"
    "v5_cuda_aware_mpi/include/layers.hpp"
    "v5_cuda_aware_mpi/src/main.cpp" # Anticipated main MPI driver
    "v5_cuda_aware_mpi/src/alexnet_mpi_cuda.cu" # Anticipated orchestrator (modified V4)
    "v5_cuda_aware_mpi/src/layers_cuda.cu" # Likely reused CUDA kernels
)

# Clear the output file
> "$OUTPUT_FILE"

echo "Collecting project files into $OUTPUT_FILE..."
echo "NOTE: Files listed as 'Not Found' have not been created yet."
echo "" >> "$OUTPUT_FILE" # Start with a blank line

# Loop through the files and append them if they exist
for relative_path in "${FILES_TO_COLLECT[@]}"; do
    full_path="$ROOT_DIR/$PROJECT_BASE/$relative_path"

    printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
    printf "\n\n=== FILE: %s ===\n\n" "$PROJECT_BASE/$relative_path" >> "$OUTPUT_FILE"

    if [[ -f "$full_path" ]]; then
        echo "Appending: $PROJECT_BASE/$relative_path"
        cat "$full_path" >> "$OUTPUT_FILE"
    else
        echo "*** File Not Found ***" >> "$OUTPUT_FILE"
        # Only print warning to terminal, not error, as this is expected for V4/V5 initially
        echo "Warning: Could not find $full_path (Expected for future versions?)" >&2
    fi
    printf "\n\n" >> "$OUTPUT_FILE"
done

printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Done. Collection complete in $OUTPUT_FILE"

exit 0