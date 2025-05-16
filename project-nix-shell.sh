#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CACHE_DIR_NAME="nix_binary_cache_local"
LOCAL_CACHE_FULL_PATH="$PROJECT_ROOT/$LOCAL_CACHE_DIR_NAME"
NIX_SHELL_FILE="$PROJECT_ROOT/shell.nix"

TEST_SCRIPTS_TO_RUN=(
    "scripts/0_run_final_project.sh"
    "scripts/1_final_unique_machine.sh"
    "scripts/2_final_multi_machine.sh"
)

NIX_SHELL_OPTIONS=()

# --- Local Cache Setup ---
if [ -d "$LOCAL_CACHE_FULL_PATH" ] && [ -f "$LOCAL_CACHE_FULL_PATH/nix-cache-info" ]; then
    echo "[INFO] project-nix-shell: Local binary cache found: $LOCAL_CACHE_FULL_PATH"
    NIX_SHELL_OPTIONS+=(
        "--option" "substituters" "file://$LOCAL_CACHE_FULL_PATH https://cache.nixos.org"
        "--option" "trusted-public-keys" "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
    )
else
    echo "[INFO] project-nix-shell: Local binary cache NOT found or invalid. Using default substituters."
fi
NIX_SHELL_OPTIONS+=("--extra-experimental-features" "nix-command flakes")

# --- Construct the command to run inside nix-shell ---
COMMAND_TO_EXECUTE_IN_SHELL_RAW=""
scripts_to_run_count=0

if [ $# -eq 0 ]; then # Only build auto-run command if NO arguments were given
    COMMAND_TO_EXECUTE_IN_SHELL_RAW="set -euo pipefail;" # Start with good flags for the inner script block
    for script_rel_path in "${TEST_SCRIPTS_TO_RUN[@]}"; do
        script_abs_for_check="$PROJECT_ROOT/$script_rel_path"
        script_path_for_nix_run="$script_rel_path"

        echo "[DEBUG] project-nix-shell: Checking for script existence at: $script_abs_for_check"
        if [ -f "$script_abs_for_check" ]; then
            if [ ! -x "$script_abs_for_check" ]; then
                echo "[WARN] project-nix-shell: Test script '$script_rel_path' not executable. Attempting chmod +x..."
                chmod +x "$script_abs_for_check" || echo "[ERROR] project-nix-shell: Failed to make '$script_rel_path' executable."
            fi
            
            # Use a subshell for each script to provide better isolation
            # Ensure paths are quoted. Using printf for the echo part is safer.
            escaped_script_rel_path_for_printf="${script_rel_path//\'/\'\\\'\'}"
            escaped_script_path_for_nix_run="${script_path_for_nix_run//\'/\'\\\'\'}"

            COMMAND_TO_EXECUTE_IN_SHELL_RAW+="( printf '\n\n--- Running Test Script: %s ---\n\n' '${escaped_script_rel_path_for_printf}'; \"${escaped_script_path_for_nix_run}\" ) && "
            scripts_to_run_count=$((scripts_to_run_count + 1))
        else
            echo "[WARN] project-nix-shell: Test script '$script_rel_path' NOT FOUND at '$script_abs_for_check'. Skipping."
        fi
    done
    # Remove trailing " && " if any scripts were added
    if [ "$scripts_to_run_count" -gt 0 ]; then
        COMMAND_TO_EXECUTE_IN_SHELL_RAW="${COMMAND_TO_EXECUTE_IN_SHELL_RAW% && }"
    fi
fi

# Escape single quotes in the entire command string for bash -c '...'
COMMAND_TO_EXECUTE_IN_SHELL_ESCAPED="${COMMAND_TO_EXECUTE_IN_SHELL_RAW//\'/\'\\\'\'}"

# --- Decide how to invoke nix-shell ---
if [ $# -eq 0 ] && [ "$scripts_to_run_count" -gt 0 ]; then
    echo "[INFO] project-nix-shell: Auto-running test script sequence."
    NIX_SHELL_OPTIONS+=("--run" "bash -c '$COMMAND_TO_EXECUTE_IN_SHELL_ESCAPED'")
elif [ $# -eq 0 ]; then
    echo "[INFO] project-nix-shell: No scripts to auto-run or none found. Entering interactive shell."
else
    echo "[INFO] project-nix-shell: Arguments provided. Passing to nix-shell: $@"
    NIX_SHELL_OPTIONS+=("$@")
fi

echo "[INFO] project-nix-shell: Executing: nix-shell $NIX_SHELL_FILE ${NIX_SHELL_OPTIONS[*]}"
if [ $# -eq 0 ] && [ "$scripts_to_run_count" -gt 0 ]; then
    echo "[INFO] project-nix-shell: Inner command for bash -c (raw, before final quoting for 'bash -c'):"
    echo "----------------------------------------------------"
    echo "$COMMAND_TO_EXECUTE_IN_SHELL_RAW"
    echo "----------------------------------------------------"
fi
echo "--------------------------------------------------------------------------------"
exec nix-shell "$NIX_SHELL_FILE" "${NIX_SHELL_OPTIONS[@]}"