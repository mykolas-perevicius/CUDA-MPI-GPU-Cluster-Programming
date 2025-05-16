#!/usr/bin/env bash
set -euo pipefail

# This script builds all project executables and stores them locally in PREBUILT_DIR_NAME.
# It directly invokes nix-shell to ensure it runs within the correct build environment.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREBUILT_DIR_NAME="prebuilt_executables_local" # Binaries stored here, gitignored
PREBUILT_FULL_PATH="$PROJECT_ROOT/$PREBUILT_DIR_NAME"
FP_SUBDIR_NAME="final_project" # Name of the final_project directory
NIX_SHELL_FILE_ABS="$PROJECT_ROOT/shell.nix" # Absolute path to shell.nix

# --- Construct the multi-line command string to be executed by bash -c ---
# Variables from this outer script ($PROJECT_ROOT, $PREBUILT_FULL_PATH, $FP_SUBDIR_NAME)
# are expanded by THIS script when the BUILD_COMMAND_STRING is defined (because of unquoted EOF).
# Variables for the INNER shell (like \$DETECTED_GPU_ARCH_FLAGS) must be escaped with \.
BUILD_COMMAND_STRING=$(cat <<EOF
set -euo pipefail # Strict mode for the inner script

echo "[BUILD_BINARIES_SCRIPT_INNER] Starting builds within Nix environment..."
echo "[BUILD_BINARIES_SCRIPT_INNER] Current directory: \$(pwd)" # Should be PROJECT_ROOT

# Source common utilities to get GPU detection logic
# Path is relative to PROJECT_ROOT, which is PWD for "nix-shell --run"
COMMON_UTILS_SCRIPT="scripts/common_test_utils.sh"
if [ -f "\$COMMON_UTILS_SCRIPT" ]; then
    source "\$COMMON_UTILS_SCRIPT" || { echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Failed to source '\$COMMON_UTILS_SCRIPT'"; exit 1; }
    # Explicitly call the detection function if it's not auto-called on source
    # This sets DETECTED_GPU_ARCH_FLAGS in this subshell's environment
    if type -t _detect_and_set_cuda_arch_flags &> /dev/null; then
        _detect_and_set_cuda_arch_flags
    else
        echo "[BUILD_BINARIES_SCRIPT_INNER_WARN] _detect_and_set_cuda_arch_flags function not found after sourcing. GPU flags may be incorrect."
    fi
    echo "[BUILD_BINARIES_SCRIPT_INNER] DETECTED_GPU_ARCH_FLAGS is: '\$DETECTED_GPU_ARCH_FLAGS'"
else
    echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] '\$COMMON_UTILS_SCRIPT' not found. Cannot determine GPU flags automatically."
    # Optionally set a fallback or exit:
    # export DETECTED_GPU_ARCH_FLAGS="-gencode arch=compute_50,code=sm_50" # Fallback example
    # exit 1
fi

# Create the main directory for prebuilt executables
mkdir -p "${PREBUILT_FULL_PATH}" # Outer script variable expansion
echo "[BUILD_BINARIES_SCRIPT_INNER] Prebuilt executables will be stored in: ${PREBUILT_FULL_PATH}"

# Define project variants and their locations
declare -A project_variants_map
project_variants_map["v1_serial"]="V1 Serial"
project_variants_map["v2_mpi_only/2.1_broadcast_all"]="V2 Broadcast"
project_variants_map["v2_mpi_only/2.2_scatter_halo"]="V2 ScatterHalo"
project_variants_map["v3_cuda_only"]="V3 CUDA"
project_variants_map["v4_mpi_cuda"]="V4 MPI+CUDA"
# project_variants_map["v5_cuda_aware_mpi"]="V5 CUDA-Aware MPI" # Add V5 when ready

for variant_subdir_key in "\${!project_variants_map[@]}"; do
    variant_display_name="\${project_variants_map[\$variant_subdir_key]}"
    # Path to the source code of the current variant
    variant_source_path="${PROJECT_ROOT}/${FP_SUBDIR_NAME}/\${variant_subdir_key}" # Outer script variables expanded

    echo ""
    echo "[BUILD_BINARIES_SCRIPT_INNER] Processing: \$variant_display_name (Source: \$variant_source_path)"

    if [ ! -d "\$variant_source_path" ]; then
        echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Directory not found: \$variant_source_path. Skipping."
        continue
    fi

    # Perform build in a subshell to isolate cd and environment changes per variant
    (
        cd "\$variant_source_path" || { echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Failed to cd to '\$variant_source_path'"; exit 1; }
        echo "[BUILD_BINARIES_SCRIPT_INNER] Building in: \$(pwd)"

        current_make_command="make"
        # Use DETECTED_GPU_ARCH_FLAGS if it's a CUDA build and the variable is set
        if [[ "\$variant_display_name" == *"CUDA"* && -n "\$DETECTED_GPU_ARCH_FLAGS" ]]; then
            current_make_command="make GPU_ARCH_FLAGS=\"\$DETECTED_GPU_ARCH_FLAGS\""
        fi
        echo "[BUILD_BINARIES_SCRIPT_INNER] Make command: \$current_make_command"

        # Clean first (ignore errors from clean)
        eval "\$current_make_command clean" > /dev/null 2>&1 || echo "[BUILD_BINARIES_SCRIPT_INNER_INFO] 'make clean' for \$variant_display_name had issues, proceeding."
        
        # Build
        if eval "\$current_make_command"; then
            if [ -f "./template" ]; then
                echo "[BUILD_BINARIES_SCRIPT_INNER] Build SUCCEEDED for: \$variant_display_name"
                
                # Determine destination path for the executable
                # Output structure: PREBUILT_FULL_PATH/v1_serial/template, PREBUILT_FULL_PATH/v2_mpi_only/2.1_broadcast_all/template, etc.
                destination_leaf_dir="\$(basename "\$variant_subdir_key")" # e.g., v1_serial, 2.1_broadcast_all
                destination_full_exe_path="${PREBUILT_FULL_PATH}/\$destination_leaf_dir/template" # Outer var PREBUILT_FULL_PATH
                
                if [[ "\$variant_subdir_key" == *"v2_mpi_only"* ]]; then
                    destination_full_exe_path="${PREBUILT_FULL_PATH}/v2_mpi_only/\$destination_leaf_dir/template"
                fi
                
                mkdir -p "\$(dirname "\$destination_full_exe_path")"
                cp "./template" "\$destination_full_exe_path"
                echo "[BUILD_BINARIES_SCRIPT_INNER] Copied './template' to '\$destination_full_exe_path'"
            else
                echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Build for \$variant_display_name reported success, but './template' executable was NOT FOUND in \$(pwd)."
            fi
        else
            echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Build FAILED for: \$variant_display_name in \$(pwd). Check 'make' output above."
        fi
    ) # End of subshell for this variant's build
    
    # Check subshell exit status
    subshell_exit_status=\$?
    if [ \$subshell_exit_status -ne 0 ]; then
        echo "[BUILD_BINARIES_SCRIPT_INNER_ERROR] Processing/build for \$variant_display_name FAILED with exit code \$subshell_exit_status."
        # Decide if you want to exit the whole script or continue with other variants
        # exit \$subshell_exit_status # To stop on first failure
    fi
done

echo ""
echo "[BUILD_BINARIES_SCRIPT_INNER] All specified variants processed."
echo "[BUILD_BINARIES_SCRIPT_INNER] Local binaries build process finished."
EOF
) # End of BUILD_COMMAND_STRING heredoc

# --- Main Script Logic (build_local_binaries.sh - Outer Script) ---
echo "Starting main script: scripts/build_local_binaries.sh"
echo "Project Root: $PROJECT_ROOT"
echo "Nix Shell File: $NIX_SHELL_FILE_ABS"
echo "Executables will be stored in: $PREBUILT_FULL_PATH"

# Ensure prebuilt_executables_local directory is gitignored
GITIGNORE_FILE="$PROJECT_ROOT/.gitignore"
PREBUILT_GITIGNORE_ENTRY="$PREBUILT_DIR_NAME/" # e.g., prebuilt_executables_local/
if ! grep -qF -- "$PREBUILT_GITIGNORE_ENTRY" "$GITIGNORE_FILE" 2>/dev/null; then
    echo "[INFO] Adding '$PREBUILT_GITIGNORE_ENTRY' to $GITIGNORE_FILE."
    # Append with newline safety
    if [ -s "$GITIGNORE_FILE" ] && [ "$(tail -c1 "$GITIGNORE_FILE")" != $'\n' ]; then echo "" >> "$GITIGNORE_FILE"; fi
    echo "$PREBUILT_GITIGNORE_ENTRY" >> "$GITIGNORE_FILE"
fi

# Prepare Nix shell options, including local cache if available
NIX_INVOCATION_OPTIONS=("--extra-experimental-features" "nix-command flakes")
LOCAL_CACHE_CHECK_PATH="$PROJECT_ROOT/nix_binary_cache_local" # Cache name from previous discussions
if [ -d "$LOCAL_CACHE_CHECK_PATH" ] && [ -f "$LOCAL_CACHE_CHECK_PATH/nix-cache-info" ]; then
    echo "[INFO] Using local binary cache for Nix shell: $LOCAL_CACHE_CHECK_PATH"
    NIX_INVOCATION_OPTIONS+=(
        "--option" "substituters" "file://$LOCAL_CACHE_CHECK_PATH https://cache.nixos.org"
        "--option" "trusted-public-keys" "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    )
else
    echo "[INFO] Local binary cache ('$LOCAL_CACHE_CHECK_PATH') not found or invalid. Using default Nix substituters."
fi

# Escape the entire BUILD_COMMAND_STRING for safe passing to bash -c '...'
# ' -> '\''
ESCAPED_BUILD_COMMAND_STRING="${BUILD_COMMAND_STRING//\'/\'\\\'\'}"

echo "[INFO] Attempting to execute build logic directly within Nix shell..."
echo "[INFO] Command for nix-shell --run: bash -c '$ESCAPED_BUILD_COMMAND_STRING'"
echo "--------------------------------------------------------------------------------"

# Execute the build logic inside nix-shell
nix-shell "$NIX_SHELL_FILE_ABS" "${NIX_INVOCATION_OPTIONS[@]}" --run "bash -c '$ESCAPED_BUILD_COMMAND_STRING'"
script_exit_code=$?

echo "--------------------------------------------------------------------------------"
if [ $script_exit_code -eq 0 ]; then
    echo "[SUCCESS] Main script 'build_local_binaries.sh' completed. Binaries should be in $PREBUILT_FULL_PATH"
else
    echo "[ERROR] Main script 'build_local_binaries.sh' encountered errors (exit code: $script_exit_code)."
fi

exit $script_exit_code