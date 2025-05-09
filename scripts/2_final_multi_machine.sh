#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# --- Configuration (USER MUST EDIT THIS SECTION) ---
# ==============================================================================

# Define your participating hosts here.
# Format for each entry: "username@hostname_or_ip cuda_arch_code"
# - username: Login username for the host.
# - hostname_or_ip: Reachable hostname or IP address of the host.
# - cuda_arch_code: The CUDA compute capability code (e.g., 50 for sm_50, 86 for sm_86).
#
# The FIRST host in this list will be treated as the "master" node,
# from which mpirun commands are initiated and where this script should be run.
HOSTS_INFO=(
    "myko@192.168.1.158 50"  # Master: 'nixos' (Quadro M1200, sm_50) - REPLACE IP IF DIFFERENT
    "myko@192.168.1.97 50"   # Worker: 'laptopB' (Quadro M1200, sm_50)
    # To add your PC later (example):
    # "your_pc_user@<PC_WSL_IP_ADDRESS> 86" # Worker PC (RTX 3090, sm_86)
)

# Set to "true" if your project directory (defined by $ROOT below) is on a
# shared filesystem (e.g., NFS) accessible at the EXACT SAME PATH on all hosts.
# If "false", the script will attempt to rsync the project from master to workers.
USE_SHARED_FILESYSTEM=false

# --- Project & Script Globals (Usually no need to edit below this line) ---
DEFAULT_USER=$(whoami)
MASTER_HOSTNAME_SHORT=$(hostname -s)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" # Project root (e.g., CUDA-MPI-GPU-Cluster-Programming)
FP_DIR="$ROOT_DIR/final_project" # final_project directory
LOGS_BASE="$FP_DIR/logs"
mkdir -p "$LOGS_BASE"

SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_ID="multirun_${SESSION_TIMESTAMP}_${MASTER_HOSTNAME_SHORT}"
SESSION_LOG_DIR="$LOGS_BASE/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"

CSV_OUTPUT_FILE="$LOGS_BASE/summary_multi_${SESSION_TIMESTAMP}.csv"
GIT_COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

PROJECT_VARIANT_TO_RUN="V4 MPI+CUDA"
PROJECT_SUBDIR="v4_mpi_cuda"
EXECUTABLE_NAME="template"
# Absolute path to the executable, assuming script is in $ROOT_DIR/scripts
# This path MUST be valid on the master node for building and on ALL nodes for execution if not using a shared FS.
TARGET_EXECUTABLE_FULL_PATH="$FP_DIR/$PROJECT_SUBDIR/$EXECUTABLE_NAME"

# Variables to store parsed host info (used by parse_host_info)
_PARSED_USER=""
_PARSED_HOST=""
_PARSED_ARCH_CODE=""

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SESSION_LOG_DIR/main_orchestration.log"
}

# Parses "user@host arch" string into global _PARSED_ vars
parse_host_info_entry() {
    local full_entry="$1"
    local user_at_host="${full_entry%% *}"
    _PARSED_ARCH_CODE="${full_entry##* }"

    if [[ "$user_at_host" == *"@"* ]]; then
        _PARSED_USER="${user_at_host%%@*}"
        _PARSED_HOST="${user_at_host##*@}"
    else # No username specified, use default user and assume entry is just hostname/IP
        _PARSED_USER="$DEFAULT_USER"
        _PARSED_HOST="$user_at_host"
    fi
}

# ==============================================================================
# --- Phase 1: SSH Setup ---
# ==============================================================================
setup_passwordless_ssh() {
    log_message "--- Phase 1: Checking/Setting up Passwordless SSH ---"
    if [[ ! -f "$HOME/.ssh/id_rsa.pub" ]]; then
        log_message "No local SSH key found ($HOME/.ssh/id_rsa.pub). Generating one..."
        ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"
        log_message "SSH key generated."
    else
        log_message "Local SSH key found ($HOME/.ssh/id_rsa.pub)."
    fi

    parse_host_info_entry "${HOSTS_INFO[0]}"
    local master_actual_user=$_PARSED_USER
    local master_actual_host=$_PARSED_HOST

    for i in "${!HOSTS_INFO[@]}"; do
        parse_host_info_entry "${HOSTS_INFO[$i]}"
        local current_target_user=$_PARSED_USER
        local current_target_host=$_PARSED_HOST
        local current_target_ssh_alias="$current_target_user@$current_target_host"

        log_message "Checking SSH to $current_target_ssh_alias from master ($master_actual_user@$MASTER_HOSTNAME_SHORT)..."

        # Attempt a non-interactive SSH command
        if ssh -o PasswordAuthentication=no -o BatchMode=yes "$current_target_ssh_alias" "exit" &>/dev/null; then
            log_message "SUCCESS: Passwordless SSH to $current_target_ssh_alias already configured."
        else
            log_message "INFO: Passwordless SSH to $current_target_ssh_alias not working or requires confirmation."
            log_message "Attempting 'ssh-copy-id $current_target_ssh_alias'. You may be prompted for the password for '$current_target_user' on '$current_target_host'."
            if ssh-copy-id "$current_target_ssh_alias"; then
                log_message "SUCCESS: 'ssh-copy-id $current_target_ssh_alias' completed."
                log_message "Verifying new passwordless SSH connection to $current_target_ssh_alias..."
                if ssh -o PasswordAuthentication=no -o BatchMode=yes "$current_target_ssh_alias" "exit" &>/dev/null; then
                    log_message "SUCCESS: Passwordless SSH to $current_target_ssh_alias confirmed."
                else
                    log_message "ERROR: Passwordless SSH to $current_target_ssh_alias still not working after ssh-copy-id. Please debug manually."
                    exit 1
                fi
            else
                log_message "ERROR: 'ssh-copy-id $current_target_ssh_alias' failed. Please set up passwordless SSH manually from '$master_actual_user@$MASTER_HOSTNAME_SHORT' to '$current_target_ssh_alias'."
                exit 1
            fi
        fi
    done
    log_message "--- SSH Setup Phase Completed ---"
}

# ==============================================================================
# --- Phase 2: Build Project (Fat Binary for Target Architectures) ---
# ==============================================================================
build_project_for_multiple_architectures() {
    log_message "--- Phase 2: Building Project ($PROJECT_VARIANT_TO_RUN) for Target Architectures ---"
    cd "$FP_DIR/$PROJECT_SUBDIR"

    local arch_flags_for_make_build=""
    declare -A unique_arch_codes_map # Use associative array for uniqueness
    for host_entry_build in "${HOSTS_INFO[@]}"; do
        parse_host_info_entry "$host_entry_build"
        unique_arch_codes_map["$_PARSED_ARCH_CODE"]=1
    done

    for code in "${!unique_arch_codes_map[@]}"; do
        arch_flags_for_make_build+="-gencode arch=compute_${code},code=sm_${code} -gencode arch=compute_${code},code=compute_${code} "
    done

    log_message "Compiling for unique CUDA compute capabilities: ${!unique_arch_codes_map[*]}"
    log_message "Effective architecture flags for make: $arch_flags_for_make_build"

    local make_log_name="make_${PROJECT_SUBDIR}_multi_arch.log"
    local make_log_path="$SESSION_LOG_DIR/$make_log_name"
    touch "$make_log_path" && > "$make_log_path" # Clear log

    log_message "Executing build (Log: $SESSION_ID/$make_log_name)..."
    # Ensure V4 Makefile is used with the correct variable name for arch flags
    if make -C "$FP_DIR/$PROJECT_SUBDIR" HOST_CUDA_ARCH_FLAGS="$arch_flags_for_make_build" clean >> "$make_log_path" 2>&1 && \
       make -C "$FP_DIR/$PROJECT_SUBDIR" HOST_CUDA_ARCH_FLAGS="$arch_flags_for_make_build" >> "$make_log_path" 2>&1; then
        if [[ -f "$TARGET_EXECUTABLE_FULL_PATH" ]]; then
            log_message "[✔ Build Succeeded: $TARGET_EXECUTABLE_FULL_PATH found]"
            log_message "--- Build Phase Completed ---"
            return 0
        else
            log_message "[✘ Build Failed: 'make' reported success BUT $TARGET_EXECUTABLE_FULL_PATH is MISSING. Check $make_log_name]"
            exit 1
        fi
    else
        log_message "[✘ Build Failed: 'make' command failed. Check $make_log_name]"
        exit 1
    fi
}

# ==============================================================================
# --- Phase 3: Synchronize Project Files (if not using shared FS) ---
# ==============================================================================
synchronize_project_files_to_workers() {
    if [[ "$USE_SHARED_FILESYSTEM" == "true" ]]; then
        log_message "--- Phase 3: Shared filesystem enabled. Skipping project rsync. ---"
        return 0
    fi

    log_message "--- Phase 3: Synchronizing Project Files to Worker Nodes ---"
    # Master node is HOSTS_INFO[0]. We sync from master to all other nodes.
    # The executable and data files must be at $TARGET_EXECUTABLE_FULL_PATH and $FP_DIR/data respectively.

    for i in $(seq 1 $((${#HOSTS_INFO[@]} - 1)) ); do # Iterate from the second host onwards
        parse_host_info_entry "${HOSTS_INFO[$i]}"
        local current_target_user=$_PARSED_USER
        local current_target_host=$_PARSED_HOST
        local current_target_ssh_alias="$current_target_user@$current_target_host"

        log_message "Syncing project from master to $current_target_ssh_alias..."
        # Ensure the root directory structure exists on the remote host.
        # rsync works relative to $ROOT_DIR.
        ssh "$current_target_ssh_alias" "mkdir -p \"$ROOT_DIR\""
        log_message "Attempting rsync of $ROOT_DIR/ to $current_target_ssh_alias:$ROOT_DIR/"

        # Sync the entire project directory. This ensures the executable and any data files are present.
        # Using --checksum can be more reliable than default timestamp/size for verifying changes.
        if rsync -avz --delete --checksum "$ROOT_DIR/" "$current_target_ssh_alias:$ROOT_DIR/"; then
            log_message "SUCCESS: Sync to $current_target_ssh_alias succeeded."
        else
            log_message "ERROR: rsync to $current_target_ssh_alias failed. Ensure $ROOT_DIR exists on remote or paths are correct."
            exit 1
        fi
    done
    log_message "--- File Synchronization Phase Completed ---"
}

# ==============================================================================
# --- Phase 4: Create MPI Hostfile ---
# ==============================================================================
create_mpi_hostfile_for_run() {
    log_message "--- Phase 4: Creating MPI Hostfile ---"
    MPI_HOSTFILE_PATH="$SESSION_LOG_DIR/mpi_cluster_hosts.txt"
    rm -f "$MPI_HOSTFILE_PATH" # Clear if exists

    for host_entry_run in "${HOSTS_INFO[@]}"; do
        parse_host_info_entry "$host_entry_run"
        # For this project, typically 1 GPU per machine, so 1 slot.
        # If a machine has more GPUs and you want to run more ranks there, adjust 'slots'.
        echo "$_PARSED_HOST slots=1" >> "$MPI_HOSTFILE_PATH"
    done
    log_message "MPI Hostfile created at: $MPI_HOSTFILE_PATH"
    log_message "Hostfile Contents:"
    cat "$MPI_HOSTFILE_PATH" | tee -a "$SESSION_LOG_DIR/main_orchestration.log"
    echo "" # Newline
    log_message "--- MPI Hostfile Creation Phase Completed ---"
}

# ==============================================================================
# --- Phase 5: Run MPI Application ---
# ==============================================================================
run_distributed_mpi_application() {
    log_message "--- Phase 5: Running Distributed MPI Application ($PROJECT_VARIANT_TO_RUN) ---"
    # No need to cd here, $TARGET_EXECUTABLE_FULL_PATH is absolute.

    local num_processes_to_run="${#HOSTS_INFO[@]}"
    local run_log_name="run_${PROJECT_SUBDIR}_multi_np${num_processes_to_run}.log"
    local run_log_path="$SESSION_LOG_DIR/$run_log_name"
    touch "$run_log_path" && > "$run_log_path" # Clear log

    # Prepare mpirun command.
    # Add --prefix $NIX_STORE_PATH_OF_OPENMPI if OpenMPI is not in default remote path and -x PATH doesn't work.
    # Add -x LD_LIBRARY_PATH if remote nodes need it for CUDA libs.
    # For NixOS, if OpenMPI and CUDA libs are in system profile on all nodes, this might be simpler.
    local mpi_command_to_run="mpirun -np $num_processes_to_run --hostfile $MPI_HOSTFILE_PATH --report-bindings --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 $TARGET_EXECUTABLE_FULL_PATH"
    # The --mca options try to prevent MPI from using loopback or docker interfaces. Adjust if your primary network interface needs to be specified.

    log_message "Executing on master node ($MASTER_HOSTNAME_SHORT): $mpi_command_to_run"
    log_message "Output will be logged to: $SESSION_ID/$run_log_name"

    local run_cmd_succeeded_bool_mpi=false
    local run_message_mpi=""
    local parsed_time_numeric_mpi=""
    local parsed_shape_mpi="–"
    local parsed_first5_mpi="–"
    local overall_status_symbol_mpi="✘"
    local overall_status_message_mpi="Not Run"


    # Execute from the directory containing the script to ensure relative paths for logs etc. are fine
    # but the executable path is absolute.
    if eval "$mpi_command_to_run" >> "$run_log_path" 2>&1; then
        log_message "[✔ MPI Run Command Succeeded]"
        run_cmd_succeeded_bool_mpi=true
        run_message_mpi="Run OK"
        overall_status_symbol_mpi="✔"
        overall_status_message_mpi="✔"

        # Attempt to parse output (assuming V4 output format is fixed in its main.cpp)
        parsed_time_str_mpi="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$run_log_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
        if [[ "$parsed_time_str_mpi" != "–" ]]; then parsed_time_numeric_mpi="${parsed_time_str_mpi// ms/}"; fi
        
        parsed_shape_mpi="$(grep -m1 '^Final Output Shape:' "$run_log_path" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
        if [[ "$parsed_shape_mpi" == "–" ]]; then # Fallback if main V4 output is not fixed yet
             parsed_shape_mpi="$(grep -m1 '^shape:' "$run_log_path" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
        fi

        parsed_first5_mpi="$(grep -m1 '^Final Output (first 10 values):' "$run_log_path" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
         if [[ "$parsed_first5_mpi" == "–" ]]; then # Fallback
             parsed_first5_mpi="$(grep -m1 '^Sample values:' "$run_log_path" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
        fi


        if [[ "$parsed_shape_mpi" == "–" || "$parsed_first5_mpi" == "–" ]]; then
            log_message "[⚠ Warning: MPI run succeeded but failed to parse critical output (shape/sample) from $run_log_name]"
            overall_status_symbol_mpi="⚠"
            overall_status_message_mpi="⚠ (parse err)"
        fi
        if [[ "$parsed_time_numeric_mpi" == "" && "$run_cmd_succeeded_bool_mpi" == true ]]; then
            log_message "[ℹ Info: Could not parse execution time from $run_log_name]"
        fi
    else
        local exit_code_mpi=$?
        log_message "[✘ MPI Run Command Failed with exit code $exit_code_mpi. Check log: $SESSION_ID/$run_log_name]"
        run_message_mpi="Runtime error (exit $exit_code_mpi)"
        overall_status_symbol_mpi="✘"
        overall_status_message_mpi="✘ (runtime err)"
    fi

    # Simplified CSV logging for this multi-run
    echo "SessionID,MachineSet,GitCommit,EntryTimestamp,ProjectVariant,NumProcesses,RunLogFile,RunCommandSucceeded,RunMessage,ExecutionTime_ms,OutputShape,OutputFirst5Values,OverallStatusSymbol,OverallStatusMessage" > "$CSV_OUTPUT_FILE" # Overwrite/Create
    
    local machine_set_identifier=""
    for host_entry_csv in "${HOSTS_INFO[@]}"; do
        parse_host_info_entry "$host_entry_csv"
        machine_set_identifier+="${_PARSED_HOST}_${_PARSED_ARCH_CODE};"
    done
    machine_set_identifier=${machine_set_identifier%;} # Remove trailing semicolon

    printf "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%s,\"%s\",%s,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"\n" \
        "$SESSION_ID" "$machine_set_identifier" "$GIT_COMMIT_HASH" "$(date --iso-8601=seconds)" \
        "$PROJECT_VARIANT_TO_RUN" "$num_processes_to_run" \
        "$SESSION_ID/$run_log_name" "$run_cmd_succeeded_bool_mpi" "$run_message_mpi" \
        "$parsed_time_numeric_mpi" "$parsed_shape_mpi" "$parsed_first5_mpi" \
        "$overall_status_symbol_mpi" "$overall_status_message_mpi" \
        >> "$CSV_OUTPUT_FILE"
    
    log_message "CSV summary for multi-run written to: $CSV_OUTPUT_FILE"
    log_message "--- MPI Application Run Phase Completed ---"
}

# ==============================================================================
# --- Main Execution Flow ---
# ==============================================================================
main() {
    # Ensure script is run from project root's 'scripts' directory or adjust ROOT_DIR definition
    if [[ ! -d "$FP_DIR" ]]; then
        echo "ERROR: final_project directory not found at $FP_DIR. Ensure script is run from the correct location or ROOT_DIR is set properly."
        exit 1
    fi
    
    # Create session log dir for main orchestration log
    echo "Multi-Machine Test Orchestration Log" > "$SESSION_LOG_DIR/main_orchestration.log"
    log_message "Starting Multi-Machine Test Script..."
    log_message "Session ID: $SESSION_ID"
    log_message "Logging to Directory: $SESSION_LOG_DIR"
    log_message "Project Root: $ROOT_DIR"
    log_message "Final Project Dir: $FP_DIR"
    log_message "Target Executable: $TARGET_EXECUTABLE_FULL_PATH"


    if [[ "${#HOSTS_INFO[@]}" -eq 0 ]]; then
        log_message "ERROR: HOSTS_INFO array is empty. Please define your hosts at the top of the script."
        exit 1
    fi
    log_message "Target hosts configured:"
    for host_entry_main in "${HOSTS_INFO[@]}"; do
        log_message "  - $host_entry_main"
    done

    setup_passwordless_ssh
    build_project_for_multiple_architectures
    synchronize_project_files_to_workers
    create_mpi_hostfile_for_run
    run_distributed_mpi_application

    log_message "--- Multi-Machine Test Script Finished ---"
    log_message "Primary run log for the MPI application: $SESSION_LOG_DIR/run_${PROJECT_SUBDIR}_multi_np${#HOSTS_INFO[@]}.log"
    log_message "Main orchestration log: $SESSION_LOG_DIR/main_orchestration.log"
    log_message "All logs for this session are in: $SESSION_LOG_DIR"
}

# --- Run Main ---
main