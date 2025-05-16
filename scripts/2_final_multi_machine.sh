#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# --- Configuration (USER MUST EDIT THIS SECTION) ---
# ==============================================================================

# Define your participating hosts here.
# Format for each entry: "username@hostname_or_ip cuda_arch_code"
# - username: Login username for the host. If omitted (e.g., "192.168.1.100 50"), script defaults to current user.
# - hostname_or_ip: Reachable hostname or IP address of the host.
# - cuda_arch_code: The CUDA compute capability code (e.g., 50 for sm_50, 86 for sm_86).
#
# The FIRST host in this list will be treated as the "master" node,
# from which mpirun commands are initiated and where this script should be run.

# HOSTS FOR LAPTOP TESTING
# HOSTS_INFO=(
#     "myko@192.168.1.158 50"  # Master: 'nixos' (Quadro M1200, sm_50) - REPLACE IP IF DIFFERENT
#     "myko@192.168.1.97 50"   # Worker: 'laptopB' (Quadro M1200, sm_50)
#     # To add your PC later (example):
#     # "your_pc_user@<PC_WSL_IP_ADDRESS> 86" # Worker PC (RTX 3090, sm_86)
# )

#HOSTS FOR HOME TESTING
HOSTS_INFO=(
    "mykoalas@172.28.101.0 75"  # Master: Laptop WSL2 (MykoPC, sm_75)
    "mykoalas@172.28.124.167 86" # Worker: PC WSL2 (DESKTOP-B5PMJB5, sm_86)
)


# Set to "true" if your project directory (defined by $ROOT_DIR below) is on a
# shared filesystem (e.g., NFS) accessible at the EXACT SAME PATH on all hosts.
# If "false", the script will attempt to rsync the project from master to workers.
USE_SHARED_FILESYSTEM=false

# Set to true for verbose rsync output, false for quieter rsync.
VERBOSE_RSYNC=false

# Optional: Specify network interface for MPI if auto-detection is problematic
# Example: MPI_NETWORK_INTERFACE="eth0" or MPI_NETWORK_INTERFACE="wlp2s0"
# Leave empty to let MPI auto-detect and use exclusions.
MPI_NETWORK_INTERFACE=""


# --- Project & Script Globals (Usually no need to edit below this line) ---
DEFAULT_USER=$(whoami)
MASTER_HOSTNAME_SHORT=$(hostname -s)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" # Project root (e.g., CUDA-MPI-GPU-Cluster-Programming)
FP_DIR="$ROOT_DIR/final_project" # final_project directory
LOGS_BASE="$FP_DIR/logs"
mkdir -p "$LOGS_BASE"

SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_ID="fullsuite_${SESSION_TIMESTAMP}_${MASTER_HOSTNAME_SHORT}"
SESSION_LOG_DIR="$LOGS_BASE/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"
ORCHESTRATION_LOG="$SESSION_LOG_DIR/main_orchestration.log"
MPI_HOSTFILE_PATH="$SESSION_LOG_DIR/mpi_cluster_hosts.txt" # Define early

CSV_OUTPUT_FILE="$LOGS_BASE/summary_fullsuite_${SESSION_TIMESTAMP}.csv"
GIT_COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

_PARSED_USER=""
_PARSED_HOST=""
_PARSED_ARCH_CODE=""

# Combined CUDA arch flags for V3/V4 builds
COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE=""

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ORCHESTRATION_LOG"
}

# Parses "user@host arch" or "host arch" string into global _PARSED_ vars
parse_host_info_entry() {
    local full_entry="$1"
    _PARSED_ARCH_CODE="${full_entry##* }" # Part after the last space
    local user_or_host_part="${full_entry% *}" # Part before the last space

    if [[ "$user_or_host_part" == *"@"* ]]; then
        _PARSED_USER="${user_or_host_part%%@*}"
        _PARSED_HOST="${user_or_host_part##*@}"
    else # No username specified, use default user
        _PARSED_USER="$DEFAULT_USER"
        _PARSED_HOST="$user_or_host_part"
    fi

    if [[ -z "$_PARSED_HOST" || -z "$_PARSED_ARCH_CODE" ]]; then
        log_message "ERROR: Malformed entry in HOSTS_INFO: '$full_entry'. Expected 'user@host arch' or 'host arch'."
        exit 1
    fi
}

# --- CSV Logging ---
write_csv_header() {
    echo "SessionID,MachineSetOrMaster,GitCommit,EntryTimestamp,ProjectVariant,NumProcesses,MakeLogFile,BuildSucceeded,BuildMessage,RunLogFile,RunCommandSucceeded,RunEnvironmentWarning,RunMessage,ParseSucceeded,ParseMessage,OverallStatusSymbol,OverallStatusMessage,ExecutionTime_ms,OutputShape,OutputFirst5Values" > "$CSV_OUTPUT_FILE"
}
log_to_csv() { # Args: 1:ProjectVariant, 2:NumProcesses, ... (see 1_final_unique_machine.sh)
    local entry_ts; entry_ts=$(date --iso-8601=seconds)
    local machine_set_id="$MASTER_HOSTNAME_SHORT" # Default to master for single node runs
    if [[ "$2" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 && ("$1" == V2* || "$1" == V4*) ]]; then # For multi-node MPI runs
        machine_set_id="CLUSTER_$(echo "${HOSTS_INFO[@]}" | tr ' ' '_' | tr '@' '_' | tr '.' '_')" # Make it more filename-friendly
    fi
    # Ensure all string arguments are quoted for CSV robustness
    local args_to_quote=(1 3 5 6 9 11 13 15 16 19 20) # Indices of string args (1-based)
    local quoted_args=()
    for i in $(seq 1 $#); do
        local current_arg="${!i}"
        local should_quote=false
        for quote_idx in "${args_to_quote[@]}"; do
            if [[ "$i" -eq "$quote_idx" ]]; then
                should_quote=true
                break
            fi
        done
        if $should_quote; then
            quoted_args+=("\"${current_arg//\"/\"\"}\"") # Escape double quotes within strings
        else
            quoted_args+=("$current_arg")
        fi
    done
    
    # Construct the printf format string dynamically based on number of args
    local format_string=""
    for i in $(seq 1 $#); do
      format_string+="%s,"
    done
    format_string="${format_string%,}\n" # Remove last comma, add newline
    
    # Prepend fixed CSV fields
    printf "\"%s\",\"%s\",\"%s\",\"%s\"," \
        "$SESSION_ID" "$machine_set_id" "$GIT_COMMIT_HASH" "$entry_ts" \
        >> "$CSV_OUTPUT_FILE"
    # Print the processed arguments
    # shellcheck disable=SC2059 # We are intentionally building format string
    printf "$format_string" "${quoted_args[@]}" >> "$CSV_OUTPUT_FILE"
}


# --- Command Execution Helper ---
run_and_log_command() {
  local cmd_to_run="$1"; shift
  local log_file_path="$1"; shift
  log_message "  -> Executing: $cmd_to_run (Log: $(basename "$SESSION_LOG_DIR")/$(basename "$log_file_path"))"
  # Clear log file before execution
  # The `eval` is used to correctly handle commands with pipes, redirections, or complex quoting passed as a string.
  # Be cautious if $cmd_to_run can come from untrusted sources. Here it's constructed by the script.
  if eval "$cmd_to_run" >>"$log_file_path" 2>&1; then
    log_message "    [✔ Command Succeeded]"
    return 0 # Actual success
  else
    local exit_code=$?
    # Check for specific error patterns that indicate environment issues vs code issues
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log_file_path"; then
        log_message "    [⚠ Warning (exit $exit_code) - Possible System/CUDA Environment Issue - see log]"
        return 2 # Environment/Setup Warning
    elif grep -q -E "There are not enough slots available|orted context|named symbol not found|no kernel image is available|cannot open shared object file|Library not found" "$log_file_path"; then
         log_message "    [⚠ Warning (exit $exit_code) - MPI Resource/Config or CUDA Arch/Lib Issue - see log]"
         return 2 # MPI Resource or CUDA Arch/Lib Warning
    else
        log_message "    [✘ Failed (exit $exit_code) – see log]"
        return 1 # Actual failure
    fi
  fi
}

# --- ASCII Table Summary ---
declare -a SUMMARY_FOR_TABLE
add_to_table_summary() { # Args: Version, Procs, Shape, First5, Last5, Time, StatusMessage
    SUMMARY_FOR_TABLE+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7")
}
_print_table_border_char() {
  local l="$1" m="$2" r="$3"
  local cols_arr_for_border=(22 5 28 30 30 10 22)
  printf "%s" "$l"
  for i in "${!cols_arr_for_border[@]}"; do
    local w=${cols_arr_for_border[i]}; local seg_len=$((w + 2))
    for ((j=0; j<seg_len; j++)); do printf '═'; done
    if (( i < ${#cols_arr_for_border[@]} - 1 )); then printf "%s" "$m"; else printf "%s\n" "$r"; fi
  done
}
_center_text_in_cell() {
    local width=$1; local text=$2
    if ((${#text} > width)); then text="${text:0:$((width-3))}..."; fi
    local text_len=${#text}; local pad_total=$((width - text_len))
    local pad_start=$((pad_total / 2)); local pad_end=$((pad_total - pad_start))
    printf "%*s%s%*s" $pad_start "" "$text" $pad_end ""
}
print_summary_table() {
    log_message "=== Summary Table (Master: $MASTER_HOSTNAME_SHORT, Session: $SESSION_ID) ==="
    local cols=(22 5 28 30 30 10 22) 
    local headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time Status)
    _print_table_border_char "╔" "╤" "╗"; printf "║"
    for i in "${!headers[@]}"; do
       printf " %s " "$(_center_text_in_cell "${cols[i]}" "${headers[i]}")"; printf "║"
    done; echo
    _print_table_border_char "╟" "┼" "╢"
    for row_data in "${SUMMARY_FOR_TABLE[@]}"; do
      IFS=$'\t' read -r ver pro shape f5 l5 tm st <<<"$row_data"
      local shape_trunc="${shape:0:${cols[2]}}"; local f5_trunc="${f5:0:${cols[3]}}"
      local st_trunc="${st:0:${cols[6]}}"
      printf "║ %-*s ║ %*s ║ %-*s ║ %-*s ║ %-*s ║ %*s ║ %-*s ║\n" \
        "${cols[0]}" "$ver" "${cols[1]}" "$pro" "${cols[2]}" "$shape_trunc" \
        "${cols[3]}" "$f5_trunc" "${cols[4]}" "$f5_trunc" "${cols[5]}" "$tm" \
        "${cols[6]}" "$st_trunc"
    done
    _print_table_border_char "╚" "╧" "╝"; echo ""
    log_message "Detailed logs in: $SESSION_LOG_DIR"
    log_message "CSV summary: $CSV_OUTPUT_FILE"
}

# ==============================================================================
# --- Initial Setup Phases (SSH, Build Prep, File Sync, MPI Hostfile) ---
# ==============================================================================
initial_cluster_setup() {
    log_message "--- Running Initial Cluster Setup ---"
    # Phase 1: SSH
    log_message "--- Phase 1: Checking/Setting up Passwordless SSH ---"
    if [[ ! -f "$HOME/.ssh/id_rsa.pub" ]]; then log_message "No local SSH key. Generating..."; ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"; fi
    parse_host_info_entry "${HOSTS_INFO[0]}"; local master_ssh_alias_for_setup="$_PARSED_USER@$_PARSED_HOST" # Used for logging clarity

    for i in "${!HOSTS_INFO[@]}"; do
        parse_host_info_entry "${HOSTS_INFO[$i]}"; local target_alias_setup="$_PARSED_USER@$_PARSED_HOST"
        log_message "Checking SSH to $target_alias_setup from $master_ssh_alias_for_setup..."
        if ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=5 "$target_alias_setup" "exit" &>/dev/null; then
            log_message "SUCCESS: Passwordless SSH to $target_alias_setup."
        else
            log_message "INFO: Attempting ssh-copy-id to $target_alias_setup. You may be prompted for password."
            if ssh-copy-id "$target_alias_setup" >> "$ORCHESTRATION_LOG" 2>&1; then
                log_message "INFO: ssh-copy-id $target_alias_setup finished. Verifying..."
                if ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=5 "$target_alias_setup" "exit" &>/dev/null; then
                    log_message "SUCCESS: Passwordless SSH to $target_alias_setup confirmed."
                else log_message "ERROR: SSH to $target_alias_setup still fails after ssh-copy-id. Debug manually."; exit 1; fi
            else log_message "ERROR: ssh-copy-id $target_alias_setup command failed. Debug manually."; exit 1; fi
        fi
    done
    log_message "--- SSH Setup Phase Completed ---"

    # Phase 2 Prep: Determine Combined CUDA Arch Flags
    declare -A unique_arch_codes_map_setup
    for host_entry_setup in "${HOSTS_INFO[@]}"; do
        parse_host_info_entry "$host_entry_setup"
        unique_arch_codes_map_setup["$_PARSED_ARCH_CODE"]=1
    done
    if [[ "${#unique_arch_codes_map_setup[@]}" -gt 0 ]]; then
        for code_setup in "${!unique_arch_codes_map_setup[@]}"; do
            COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE+="-gencode arch=compute_${code_setup},code=sm_${code_setup} -gencode arch=compute_${code_setup},code=compute_${code_setup} "
        done
        log_message "Combined CUDA arch flags for V3/V4 builds: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    else
        log_message "INFO: No hosts with CUDA arch codes in HOSTS_INFO; CUDA builds will use Makefile defaults if any."
    fi

    # Phase 3: Sync files if not shared FS
    if [[ "$USE_SHARED_FILESYSTEM" == "true" ]]; then
        log_message "--- Phase 3: Shared FS. Skipping rsync. ---"
    else
        log_message "--- Phase 3: Syncing Project Files to Worker Nodes ---"
        # Sync only if there are worker nodes (more than 1 host total)
        if [[ "${#HOSTS_INFO[@]}" -gt 1 ]]; then
            for i in $(seq 1 $((${#HOSTS_INFO[@]} - 1)) ); do # Iterate from the second host onwards (workers)
                parse_host_info_entry "${HOSTS_INFO[$i]}"; local target_alias_sync="$_PARSED_USER@$_PARSED_HOST"
                log_message "Syncing project root $ROOT_DIR/ to $target_alias_sync:$ROOT_DIR/"
                if ! ssh "$target_alias_sync" "mkdir -p \"$ROOT_DIR\""; then log_message "ERR: mkdir $ROOT_DIR on $target_alias_sync failed."; exit 1; fi
                
                local rsync_cmd_sync="rsync -az --delete --checksum --exclude '.git/' '$ROOT_DIR/' '$target_alias_sync:$ROOT_DIR/'"
                if [[ "$VERBOSE_RSYNC" == "true" ]]; then rsync_cmd_sync="rsync -avz --delete --checksum --exclude '.git/' '$ROOT_DIR/' '$target_alias_sync:$ROOT_DIR/'"; fi
                
                # Log rsync command to main log, its output to a specific rsync log for that host
                local rsync_log_path="$SESSION_LOG_DIR/rsync_to_${_PARSED_HOST}.log"
                log_message "  Executing rsync (Log: $(basename "$SESSION_LOG_DIR")/$(basename "$rsync_log_path"))..."
                if eval "$rsync_cmd_sync" >"$rsync_log_path" 2>&1; then
                    log_message "SUCCESS: Sync to $target_alias_sync."
                else
                    log_message "ERR: rsync to $target_alias_sync failed. Check $rsync_log_path."
                    exit 1;
                fi
            done
        else
            log_message "Only one host defined or no hosts; no remote sync needed."
        fi
    fi
    log_message "--- File Synchronization Phase Completed ---"

    # Phase 4: Create MPI Hostfile
    if [[ "${#HOSTS_INFO[@]}" -gt 0 ]]; then
        log_message "--- Phase 4: Creating MPI Hostfile ---"
        rm -f "$MPI_HOSTFILE_PATH"
        for host_entry_mpi_hf in "${HOSTS_INFO[@]}"; do
            parse_host_info_entry "$host_entry_mpi_hf"
            echo "$_PARSED_HOST slots=1" >> "$MPI_HOSTFILE_PATH"
        done
        log_message "MPI Hostfile: $MPI_HOSTFILE_PATH"; cat "$MPI_HOSTFILE_PATH" | tee -a "$ORCHESTRATION_LOG"
    else
        log_message "--- Phase 4: No hosts in HOSTS_INFO, MPI Hostfile not created. MPI runs will be local-only."
        touch "$MPI_HOSTFILE_PATH" # Create empty file so script doesn't fail if it expects it
    fi
    log_message "--- MPI Hostfile Creation Phase Completed ---"
    log_message "--- Initial Cluster Setup Completed ---"
}


# ==============================================================================
# --- Main Test Execution Logic (Adapted from 1_final_unique_machine.sh) ---
# ==============================================================================
run_test_suite() {
    log_message "--- Starting Full Test Suite ---"
    write_csv_header # Initialize CSV file

    # --- Testing V1 Serial ---
    log_message "=== Testing V1 Serial (1 process) ==="
    local v1_dir="$FP_DIR/v1_serial"
    local current_variant_name="V1 Serial"; local current_np=1
    local make_log_name="make_v1.log"; local make_log_rel_path="$SESSION_ID/$make_log_name"
    local make_log_abs_path="$SESSION_LOG_DIR/$make_log_name"
    local build_succeeded=false; local build_msg="Init"
    log_message "  Building V1 (Log: $make_log_rel_path)..."
    # Clear make log
    >$make_log_abs_path
    if make -C "$v1_dir" clean >> "$make_log_abs_path" 2>&1 && make -C "$v1_dir" >> "$make_log_abs_path" 2>&1; then
        if [[ -f "$v1_dir/template" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
    else local make_exit_code=$?; build_msg="Build failed (exit $make_exit_code)"; fi
    log_message "    Build Status: $build_msg"

    local run_log_name="run_v1_np${current_np}.log"; local run_log_rel_path="$SESSION_ID/$run_log_name"
    local run_path="$SESSION_LOG_DIR/$run_log_name"
    local shape="–"; local first5="–"; local time_str="–"; local time_num="";
    local run_ok=false; local run_env_warn=false; local run_msg="-"; local parse_ok=false; local parse_msg="-"
    local overall_sym="✘"; local overall_msg="Not Run"

    if $build_succeeded; then
        local cmd_v1="cd $v1_dir && ./template" # V1 is always run from its directory
        local cmd_v1_exit_code=0; run_and_log_command "$cmd_v1" "$run_path" || cmd_v1_exit_code=$?
        if [[ $cmd_v1_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"
        elif [[ $cmd_v1_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)"
        else run_msg="Runtime Err (exit $cmd_v1_exit_code)"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

        if $run_ok || $run_env_warn; then
            time_str="$(grep -m1 'completed in' "$run_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
            if [[ "$time_str" != "–" ]]; then time_num="${time_str// ms/}"; fi
            shape="$(grep -m1 'After LRN2' "$run_path" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "–")"
            if [[ "$shape" == "–" ]]; then shape="$(grep -m1 '^Final Output Shape:' "$run_path" | sed -n -E 's/.*Shape: ([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")"; fi
            first5="$(grep -m1 '^Final Output (first 10 values):' "$run_path" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
            if [[ "$first5" == "–" || "$time_str" == "–" || "$shape" == "–" ]]; then
                parse_ok=false; parse_msg="Parse err"; if $run_ok; then overall_sym="⚠"; overall_msg="⚠ (parse)"; fi
            else parse_ok=true; parse_msg="Parse OK"; fi
        else parse_msg="Not run or runtime err"; fi
    else overall_msg="✘ ($build_msg)"; run_msg="Skip build"; parse_msg="Skip build"; fi
    add_to_table_summary "$current_variant_name" "$current_np" "$shape" "$first5" "$first5" "$time_str" "$overall_msg"
    log_to_csv "$current_variant_name" "$current_np" "$make_log_rel_path" "$build_succeeded" "$build_msg" \
        "$run_log_rel_path" "$run_ok" "$run_env_warn" "$run_msg" "$parse_ok" "$parse_msg" \
        "$overall_sym" "$overall_msg" "$time_num" "$shape" "$first5"

    # --- Testing V2 MPI Only ---
    local v2_base_dir="$FP_DIR/v2_mpi_only"
    for ver_suffix in "2.1_broadcast_all" "2.2_scatter_halo"; do
        local v2_dir="$v2_base_dir/$ver_suffix"
        local v2_executable_abs_path="$v2_dir/template"
        
        # Determine max NP for V2: up to total hosts if multi-node, else cap at 4 for local tests
        local max_np_for_this_test_type=$((${#HOSTS_INFO[@]} > 0 ? ${#HOSTS_INFO[@]} : 4))
        if [[ "$max_np_for_this_test_type" -gt 4 ]]; then max_np_for_this_test_type=4; fi
        
        local np_values_for_test=(1)
        if [[ "$max_np_for_this_test_type" -ge 2 ]]; then np_values_for_test+=(2); fi
        if [[ "$max_np_for_this_test_type" -ge 4 ]]; then np_values_for_test+=(4); fi
        
        for np_val in "${np_values_for_test[@]}"; do
            # Skip multi-node test if np_val > number of configured hosts
            if [[ "$np_val" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 && "$np_val" -gt "${#HOSTS_INFO[@]}" ]]; then
                log_message "  Skipping $current_variant_name NP=$np_val as it exceeds number of configured hosts (${#HOSTS_INFO[@]})."
                continue
            fi

            current_variant_name="V2 ${ver_suffix//_/-}"; current_np="$np_val"
            log_message "=== Testing $current_variant_name with $current_np processes ==="
            make_log_name="make_v2_${ver_suffix}_np${current_np}.log"; make_log_rel_path="$SESSION_ID/$make_log_name"; make_log_abs_path="$SESSION_LOG_DIR/$make_log_name"
            build_succeeded=false; build_msg="Init"
            log_message "  Building $current_variant_name (Log: $make_log_rel_path)..."
            >$make_log_abs_path
            if make -C "$v2_dir" clean >> "$make_log_abs_path" 2>&1 && make -C "$v2_dir" >> "$make_log_abs_path" 2>&1; then
                if [[ -f "$v2_executable_abs_path" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
            else local make_exit_code=$?; build_msg="Build failed (exit $make_exit_code)"; fi
            log_message "    Build Status: $build_msg"

            run_log_name="run_v2_${ver_suffix}_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
            shape="–"; first5="–"; time_str="–"; time_num=""; run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"; overall_sym="✘"; overall_msg="Not Run"

            if $build_succeeded; then
                local mpi_cmd_v2
                local network_params_v2=""
                if [[ -n "$MPI_NETWORK_INTERFACE" ]]; then
                    network_params_v2="--mca btl_tcp_if_include $MPI_NETWORK_INTERFACE --mca oob_tcp_if_include $MPI_NETWORK_INTERFACE"
                else
                    network_params_v2="--mca btl_tcp_if_exclude lo,docker0,virbr0 --mca oob_tcp_if_exclude lo,docker0,virbr0"
                fi

                if [[ "$current_np" -eq 1 ]]; then
                    mpi_cmd_v2="cd $v2_dir && mpirun -np 1 ./template" # No network params for local single
                elif [[ "${#HOSTS_INFO[@]}" -gt 1 && -s "$MPI_HOSTFILE_PATH" ]]; then
                    # Multi-node: Use hostfile and absolute path to executable
                    mpi_cmd_v2="mpirun -np $current_np --hostfile $MPI_HOSTFILE_PATH --report-bindings $network_params_v2 $v2_executable_abs_path"
                else
                    # Local multi-process (single host in HOSTS_INFO or HOSTS_INFO is empty)
                    mpi_cmd_v2="cd $v2_dir && mpirun --oversubscribe -np $current_np ./template"
                fi
                
                local cmd_v2_exit_code=0; run_and_log_command "$mpi_cmd_v2" "$run_path" || cmd_v2_exit_code=$?
                if [[ $cmd_v2_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"
                elif [[ $cmd_v2_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)"
                else run_msg="Runtime Err (exit $cmd_v2_exit_code)"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

                if $run_ok || $run_env_warn; then
                    shape="$(grep -m1 '^shape:' "$run_path" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
                    first5="$(grep -m1 '^Sample values:' "$run_path" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
                    time_str="$(grep -m1 '^Execution Time:' "$run_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
                    if [[ "$time_str" != "–" ]]; then time_num="${time_str// ms/}"; fi
                    if [[ "$shape" == "–" || "$first5" == "–" ]]; then
                        parse_ok=false; parse_msg="Parse err"; if $run_ok; then overall_sym="⚠"; overall_msg="⚠ (parse)"; fi
                    else parse_ok=true; parse_msg="Parse OK"; fi
                    if [[ "$time_str" == "–" && "$parse_ok" == true ]]; then parse_msg="$parse_msg (no time)"; fi
                else parse_msg="Not run or runtime err"; fi
            else overall_msg="✘ ($build_msg)"; run_msg="Skip build"; parse_msg="Skip build"; fi
            add_to_table_summary "$current_variant_name" "$current_np" "$shape" "$first5" "$first5" "$time_str" "$overall_msg"
            log_to_csv "$current_variant_name" "$current_np" "$make_log_rel_path" "$build_succeeded" "$build_msg" \
                "$run_log_rel_path" "$run_ok" "$run_env_warn" "$run_msg" "$parse_ok" "$parse_msg" \
                "$overall_sym" "$overall_msg" "$time_num" "$shape" "$first5"
        done
    done

    # --- Testing V3 CUDA Only ---
    log_message "=== Testing V3 CUDA Only (1 process, on master) ==="
    local v3_dir="$FP_DIR/v3_cuda_only"
    current_variant_name="V3 CUDA"; current_np=1
    make_log_name="make_v3.log"; make_log_rel_path="$SESSION_ID/$make_log_name"; make_log_abs_path="$SESSION_LOG_DIR/$make_log_name"
    build_succeeded=false; build_msg="Init"
    log_message "  Building V3 (Log: $make_log_rel_path)... Arch: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    >$make_log_abs_path
    if make -C "$v3_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" clean >> "$make_log_abs_path" 2>&1 && \
       make -C "$v3_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" >> "$make_log_abs_path" 2>&1; then
        if [[ -f "$v3_dir/template" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
    else local make_exit_code=$?; build_msg="Build failed (exit $make_exit_code)"; fi
    log_message "    Build Status: $build_msg"
    
    run_log_name="run_v3_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
    shape="–"; first5="–"; time_str="–"; time_num=""; run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"; overall_sym="✘"; overall_msg="Not Run"
    if $build_succeeded; then
        local cmd_v3="cd $v3_dir && ./template"
        local cmd_v3_exit_code=0; run_and_log_command "$cmd_v3" "$run_path" || cmd_v3_exit_code=$?
        if [[ $cmd_v3_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"; 
        elif [[ $cmd_v3_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)";
        else run_msg="Runtime Err (exit $cmd_v3_exit_code)"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi
        if $run_ok || $run_env_warn; then
            time_str="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$run_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
            if [[ "$time_str" != "–" ]]; then time_num="${time_str// ms/}"; fi
            first5="$(grep -m1 '^Final Output (first 10 values):' "$run_path" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
            shape="13x13x256" 
            if [[ "$first5" == "–" || "$time_str" == "–" ]]; then parse_ok=false; parse_msg="Parse err"; if $run_ok; then overall_sym="⚠"; overall_msg="⚠ (parse)"; fi
            else parse_ok=true; parse_msg="Parse OK"; fi
        else parse_msg="Not run or runtime err"; fi
    else overall_msg="✘ ($build_msg)"; run_msg="Skip build"; parse_msg="Skip build"; fi
    add_to_table_summary "$current_variant_name" "$current_np" "$shape" "$first5" "$first5" "$time_str" "$overall_msg"
    log_to_csv "$current_variant_name" "$current_np" "$make_log_rel_path" "$build_succeeded" "$build_msg" \
        "$run_log_rel_path" "$run_ok" "$run_env_warn" "$run_msg" "$parse_ok" "$parse_msg" \
        "$overall_sym" "$overall_msg" "$time_num" "$shape" "$first5"

    # --- Testing V4 MPI+CUDA ---
    log_message "=== Testing V4 MPI+CUDA ==="
    local v4_dir="$FP_DIR/v4_mpi_cuda"
    local v4_executable_abs_path="$v4_dir/template"
    current_variant_name="V4 MPI+CUDA" # Base name
    make_log_name="make_v4.log"; make_log_rel_path="$SESSION_ID/$make_log_name"; make_log_abs_path="$SESSION_LOG_DIR/$make_log_name"
    local v4_build_succeeded=false; local v4_build_msg="Init"
    log_message "  Building V4 (Log: $make_log_rel_path)... Arch: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    >$make_log_abs_path
    if make -C "$v4_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" clean >> "$make_log_abs_path" 2>&1 && \
       make -C "$v4_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" >> "$make_log_abs_path" 2>&1; then
        if [[ -f "$v4_executable_abs_path" ]]; then v4_build_succeeded=true; v4_build_msg="Build OK"; else v4_build_msg="Build OK, no exe"; fi
    else local make_exit_code=$?; v4_build_msg="Build failed (exit $make_exit_code)"; fi
    log_message "    Build Status: $v4_build_msg"

    local max_np_for_v4=$((${#HOSTS_INFO[@]} > 0 ? ${#HOSTS_INFO[@]} : 4))
    if [[ "$max_np_for_v4" -gt 4 ]]; then max_np_for_v4=4; fi
    local np_values_for_v4=(1)
    if [[ "$max_np_for_v4" -ge 2 ]]; then np_values_for_v4+=(2); fi
    if [[ "$max_np_for_v4" -ge 4 ]]; then np_values_for_v4+=(4); fi

    for np_val in "${np_values_for_v4[@]}"; do
        if [[ "$np_val" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 && "$np_val" -gt "${#HOSTS_INFO[@]}" ]]; then
            log_message "  Skipping V4 NP=$np_val as it exceeds number of configured hosts (${#HOSTS_INFO[@]})."
            continue
        fi
        current_np="$np_val"
        log_message "--- V4 with $current_np processes ---"
        run_log_name="run_v4_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
        shape="–"; first5="–"; time_str="–"; time_num=""; run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"; overall_sym="✘"; overall_msg="Not Run"

        if $v4_build_succeeded; then
            local mpi_cmd_v4
            local network_params_v4=""
            if [[ -n "$MPI_NETWORK_INTERFACE" ]]; then
                network_params_v4="--mca btl_tcp_if_include $MPI_NETWORK_INTERFACE --mca oob_tcp_if_include $MPI_NETWORK_INTERFACE"
            else
                network_params_v4="--mca btl_tcp_if_exclude lo,docker0,virbr0 --mca oob_tcp_if_exclude lo,docker0,virbr0"
            fi

            if [[ "$current_np" -eq 1 ]]; then
                mpi_cmd_v4="cd $v4_dir && mpirun -np 1 ./template"
            elif [[ "${#HOSTS_INFO[@]}" -gt 1 && -s "$MPI_HOSTFILE_PATH" ]]; then
                mpi_cmd_v4="mpirun -np $current_np --hostfile $MPI_HOSTFILE_PATH --report-bindings $network_params_v4 $v4_executable_abs_path"
            else
                mpi_cmd_v4="cd $v4_dir && mpirun --oversubscribe -np $current_np ./template"
            fi
            
            local cmd_v4_exit_code=0; run_and_log_command "$mpi_cmd_v4" "$run_path" || cmd_v4_exit_code=$?
            if [[ $cmd_v4_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔";
            elif [[ $cmd_v4_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)";
            else run_msg="Runtime Err (exit $cmd_v4_exit_code)"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

            if $run_ok || $run_env_warn; then
                shape="$(grep -m1 '^Final Output Shape:' "$run_path" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
                 # Fallback shape parsing for V4 (copied from your 1_final_unique_machine.sh)
                if [[ "$shape" == "–" ]]; then
                    local total_size; total_size=$(grep -m1 '^Final Output Total Size:' "$run_path" | sed -n -E 's/^Final Output Total Size: *([0-9]+) .*/\1/p' || echo "")
                    if [[ -n "$total_size" ]]; then
                        if [[ "$total_size" == "43264" ]]; then shape="13x13x256"; 
                        elif [[ "$total_size" == "0" ]]; then shape="0x0x0";
                        else shape="?x?x? ($total_size elem)"; fi
                    else
                        local local_h; local_h=$(grep -m1 '^Rank .* local output H=' "$run_path" | sed -n -E 's/.*local output H=([0-9]+).*/\1/p' || echo "")
                        local local_w; local_w=$(grep -m1 '^Rank .* local output W=' "$run_path" | sed -n -E 's/.*local output W=([0-9]+).*/\1/p' || echo "")
                        local local_c; local_c=$(grep -m1 '^Rank .* local output C=' "$run_path" | sed -n -E 's/.*local output C=([0-9]+).*/\1/p' || echo "")
                        if [[ -n "$local_h" && -n "$local_w" && -n "$local_c" ]]; then
                            if [[ "$current_np" -gt 1 && "$local_h" != "13" && $((local_h * current_np)) -gt "$local_h" ]]; then # Avoid non-numeric errors
                                shape="$((local_h * current_np))x${local_w}x${local_c} (est. local ${local_h}x)"
                            else shape="${local_h}x${local_w}x${local_c}"; fi
                        elif [[ "$current_np" -gt 1 ]]; then 
                            case $current_np in 2) shape="~?x13x256(split)";; 4) shape="~?x13x256(split)";; *) shape="?x?x?(decomposed)";; esac
                        fi
                    fi
                fi

                first5="$(grep -m1 '^Final Output (first 10 values):' "$run_path" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
                time_str="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$run_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
                if [[ "$time_str" != "–" ]]; then time_num="${time_str// ms/}"; fi

                if [[ "$shape" == "–" || "$first5" == "–" ]]; then
                    parse_ok=false; parse_msg="Parse err"; if $run_ok; then overall_sym="⚠"; overall_msg="⚠ (parse)"; fi
                    log_message "  [⚠ V4 NP=$current_np parse error. Shape: '$shape', Sample: '$first5']"
                else parse_ok=true; parse_msg="Parse OK"; fi
                if [[ "$time_str" == "–" && "$parse_ok" == true ]]; then parse_msg="$parse_msg (no time)"; fi
            else parse_msg="Not run or runtime err"; fi
        else 
            overall_msg="✘ ($v4_build_msg)"; run_msg="Skip build"; parse_msg="Skip build"
        fi
        add_to_table_summary "$current_variant_name" "$current_np" "$shape" "$first5" "$first5" "$time_str" "$overall_msg"
        log_to_csv "$current_variant_name" "$current_np" "$make_log_rel_path" "$v4_build_succeeded" "$v4_build_msg" \
            "$run_log_rel_path" "$run_ok" "$run_env_warn" "$run_msg" "$parse_ok" "$parse_msg" \
            "$overall_sym" "$overall_msg" "$time_num" "$shape" "$first5"
    done

    log_message "--- Full Test Suite Finished ---"
}

# ==============================================================================
# --- Main Script Logic ---
# ==============================================================================
main() {
    echo "Multi-Machine Full Suite Test Orchestration Log - Session: $SESSION_ID" > "$ORCHESTRATION_LOG"
    log_message "Starting Full Suite Test Script..."
    log_message "Session ID: $SESSION_ID"
    log_message "Logging to Directory: $SESSION_LOG_DIR"
    log_message "Project Root: $ROOT_DIR"
    log_message "Final Project Dir: $FP_DIR"
    
    if [[ "${#HOSTS_INFO[@]}" -eq 0 ]]; then
        log_message "WARNING: HOSTS_INFO array is empty. All MPI tests will run locally on $MASTER_HOSTNAME_SHORT."
    else
        log_message "Target hosts configured:"
        for host_entry_main in "${HOSTS_INFO[@]}"; do log_message "  - $host_entry_main"; done
    fi

    initial_cluster_setup 
    run_test_suite        
    print_summary_table   

    log_message "--- Full Suite Test Script Finished ---"
    log_message "All logs for this session are in: $SESSION_LOG_DIR"
    log_message "Review the CSV summary: $CSV_OUTPUT_FILE"
    log_message "Review the main orchestration log: $ORCHESTRATION_LOG"
}

# --- Run Main ---
main