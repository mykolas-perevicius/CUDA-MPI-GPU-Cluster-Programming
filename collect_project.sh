#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root directory:
# ~/CS485/CUDA-MPI-GPU-Cluster-Programming$ bash collect_project.sh [v1] [v2.2] [v3] [v4] [v5] ...

ROOT_DIR="$(pwd)"
OUTPUT_FILE="project.txt"
PROJECT_BASE="final_project"

# Array of requested versions from command line args
requested_versions=("$@")

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
    "v4_mpi_cuda/src/main_mpi_cuda.cpp" # Corrected name
    "v4_mpi_cuda/src/alexnet_mpi_cuda.cu"
    "v4_mpi_cuda/src/layers_mpi_cuda.cu" # Corrected name

    # V5 CUDA-Aware MPI - Optimization (Include if exists)
    "v5_cuda_aware_mpi/Makefile"
    "v5_cuda_aware_mpi/include/alexnet.hpp"
    "v5_cuda_aware_mpi/include/layers.hpp"
    "v5_cuda_aware_mpi/src/main.cpp" # Anticipated main MPI driver (adjust name if needed)
    "v5_cuda_aware_mpi/src/alexnet_mpi_cuda.cu" # Anticipated orchestrator (modified V4)
    "v5_cuda_aware_mpi/src/layers_cuda.cu" # Likely reused CUDA kernels (adjust name if needed)
)

# Clear the output file
> "$OUTPUT_FILE"

echo "Collecting project files into $OUTPUT_FILE..."

# Check if specific versions were requested
if [ ${#requested_versions[@]} -gt 0 ]; then
    echo "Requested versions: ${requested_versions[@]}"
else
    echo "No specific versions requested, collecting all available."
fi
echo "NOTE: Files listed as 'Not Found' may be expected for un-implemented versions."
echo "" >> "$OUTPUT_FILE" # Start with a blank line

# Loop through the files and append them if they exist and match requested versions
files_appended_count=0
for relative_path in "${FILES_TO_COLLECT[@]}"; do
    # --- Filtering Logic ---
    process_this_file=false
    if [ ${#requested_versions[@]} -eq 0 ]; then
        # No versions specified, default to processing all
        process_this_file=true
    else
        # Check if the path matches any requested version prefix
        for version_request in "${requested_versions[@]}"; do
            version_path_prefix=""
            # Define the expected directory prefix for each version string
            case "$version_request" in
                v1)   version_path_prefix="v1_serial/" ;;
                # Allow requesting just 'v2' to get both, or specific sub-versions
                v2)   version_path_prefix="v2_mpi_only/" ;;
                v2.1) version_path_prefix="v2_mpi_only/2.1_broadcast_all/" ;; # Add this pattern if needed
                v2.2) version_path_prefix="v2_mpi_only/2.2_scatter_halo/" ;;
                v3)   version_path_prefix="v3_cuda_only/" ;;
                v4)   version_path_prefix="v4_mpi_cuda/" ;;
                v5)   version_path_prefix="v5_cuda_aware_mpi/" ;;
                *)
                  # Allow skipping unknown requests silently or print a warning once
                  # if [[ "$printed_unknown_warning" != "true" ]]; then
                  #   echo "Warning: Unknown version request '$version_request'. Ignoring." >&2
                  #   printed_unknown_warning="true" # Prevent repeated warnings
                  # fi
                  ;;
            esac

            # Check if the relative path starts with the determined prefix
            if [[ -n "$version_path_prefix" && "$relative_path" == "$version_path_prefix"* ]]; then
                process_this_file=true
                break # Found a match for this file, no need to check other requested versions
            fi
        done
    fi
    # --- End Filtering Logic ---

    # If the file should be processed (either all requested or version matched)
    if [[ "$process_this_file" = true ]]; then
        full_path="$ROOT_DIR/$PROJECT_BASE/$relative_path"

        printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
        printf "\n\n=== FILE: %s ===\n\n" "$PROJECT_BASE/$relative_path" >> "$OUTPUT_FILE"

        if [[ -f "$full_path" ]]; then
            echo "Appending: $PROJECT_BASE/$relative_path"
            cat "$full_path" >> "$OUTPUT_FILE"
            files_appended_count=$((files_appended_count + 1))
        else
            echo "*** File Not Found ***" >> "$OUTPUT_FILE"
            # Print warning to terminal only if no specific versions were requested OR
            # if the specific requested version's file is missing.
            if [ ${#requested_versions[@]} -eq 0 ] || [[ "$relative_path" == "$version_path_prefix"* ]]; then
                 echo "Warning: Could not find $full_path" >&2
            fi
        fi
        printf "\n\n" >> "$OUTPUT_FILE"
    fi # end if process_this_file
done

printf '=%.0s' {1..80} >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

if [ $files_appended_count -eq 0 ]; then
     echo "Warning: No files were collected. Check requested versions and file paths." >&2
     echo "(No files matched requested versions or base files are missing)" >> "$OUTPUT_FILE"
fi

echo "Done. Collection complete in $OUTPUT_FILE ($files_appended_count files appended)."

exit 0