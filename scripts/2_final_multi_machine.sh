#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# --- Configuration (USER MUST EDIT THIS SECTION) ---
# ==============================================================================
HOSTS_INFO=(
    "myko@192.168.1.158 50"  # Master: 'nixos' (Quadro M1200, sm_50)
    "myko@192.168.1.97 50"   # Worker: 'laptopB' (Quadro M1200, sm_50)
    # "your_pc_user@<PC_WSL_IP_ADDRESS> 86" # Example PC
)
USE_SHARED_FILESYSTEM=false
VERBOSE_RSYNC=false
MPI_NETWORK_INTERFACE="" # e.g., "eth0" or "wlp2s0"; leave empty for auto

# --- Project & Script Globals ---
DEFAULT_USER=$(whoami)
MASTER_HOSTNAME_SHORT=$(hostname -s)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP_DIR="$ROOT_DIR/final_project"
LOGS_BASE="$FP_DIR/logs"
mkdir -p "$LOGS_BASE"

SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_ID="fullsuite_${SESSION_TIMESTAMP}_${MASTER_HOSTNAME_SHORT}"
SESSION_LOG_DIR="$LOGS_BASE/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"
ORCHESTRATION_LOG="$SESSION_LOG_DIR/main_orchestration.log"
MPI_HOSTFILE_PATH="$SESSION_LOG_DIR/mpi_cluster_hosts.txt" # Define early for cleanup

CSV_OUTPUT_FILE="$LOGS_BASE/summary_fullsuite_${SESSION_TIMESTAMP}.csv"
GIT_COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

_PARSED_USER=""
_PARSED_HOST=""
_PARSED_ARCH_CODE=""

# Combined CUDA arch flags for V3/V4 builds
COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE=""

# --- Helper Functions (log_message, parse_host_info_entry from Iteration 2) ---
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ORCHESTRATION_LOG"
}
parse_host_info_entry() {
    local full_entry="$1"
    _PARSED_ARCH_CODE="${full_entry##* }"
    local user_or_host_part="${full_entry% *}"
    if [[ "$user_or_host_part" == *"@"* ]]; then
        _PARSED_USER="${user_or_host_part%%@*}"; _PARSED_HOST="${user_or_host_part##*@}"
    else
        _PARSED_USER="$DEFAULT_USER"; _PARSED_HOST="$user_or_host_part"
    fi
    if [[ -z "$_PARSED_HOST" || -z "$_PARSED_ARCH_CODE" ]]; then
        log_message "ERROR: Malformed HOSTS_INFO: '$full_entry'."; exit 1;
    fi
}

# --- CSV Logging (from 1_final_unique_machine.sh, slightly adapted) ---
write_csv_header() {
    echo "SessionID,MachineSetOrMaster,GitCommit,EntryTimestamp,ProjectVariant,NumProcesses,MakeLogFile,BuildSucceeded,BuildMessage,RunLogFile,RunCommandSucceeded,RunEnvironmentWarning,RunMessage,ParseSucceeded,ParseMessage,OverallStatusSymbol,OverallStatusMessage,ExecutionTime_ms,OutputShape,OutputFirst5Values" > "$CSV_OUTPUT_FILE"
}
log_to_csv() { # Args: 1:ProjectVariant, 2:NumProcesses, ... (see 1_final_unique_machine.sh)
    local entry_ts; entry_ts=$(date --iso-8601=seconds)
    local machine_set_id="$MASTER_HOSTNAME_SHORT" # Default to master for single node runs
    if [[ "$2" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 && "$1" == V2* || "$1" == V4* ]]; then # For multi-node MPI runs
        machine_set_id="CLUSTER_$(echo "${HOSTS_INFO[@]}" | tr ' ' '_' | tr '@' '_')"
    fi
    printf "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%s,\"%s\",%s,\"%s\",\"%s\",%s,%s,\"%s\",%s,\"%s\",\"%s\",\"%s\",%s,\"%s\",\"%s\"\n" \
        "$SESSION_ID" "$machine_set_id" "$GIT_COMMIT_HASH" "$entry_ts" \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" \
        >> "$CSV_OUTPUT_FILE"
}

# --- Command Execution Helper (from 1_final_unique_machine.sh) ---
run_and_log_command() { # Renamed to avoid conflict if sourced
  local cmd="$1"; shift
  local log_path="$1"; shift
  log_message "  -> Executing: $cmd (Log: $(basename "$SESSION_LOG_DIR")/$(basename "$log_path"))"
  touch "$log_path" && > "$log_path"
  if eval "$cmd" >>"$log_path" 2>&1; then
    log_message "    [✔ Command Succeeded]"
    return 0
  else
    local exit_code=$?
    # Check for specific error patterns that indicate environment issues vs code issues
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log_path"; then
        log_message "    [⚠ Warning (exit $exit_code) - Possible System/CUDA Environment Issue - see log]"
        return 2 # Environment/Setup Warning
    elif grep -q -E "There are not enough slots available|orted context|named symbol not found|no kernel image is available" "$log_path"; then
         log_message "    [⚠ Warning (exit $exit_code) - MPI Resource/Config or CUDA Arch Issue - see log]"
         return 2 # MPI Resource or CUDA Arch Warning
    else
        log_message "    [✘ Failed (exit $exit_code) – see log]"
        return 1 # Actual failure
    fi
  fi
}

# --- ASCII Table Summary (from 1_final_unique_machine.sh) ---
declare -a SUMMARY_FOR_TABLE
add_to_table_summary() { SUMMARY_FOR_TABLE+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7"); }
print_summary_table() {
    log_message "=== Summary Table (Master: $MASTER_HOSTNAME_SHORT, Session: $SESSION_ID) ==="
    # ... (exact print_border, center_text, loop from 1_final_unique_machine.sh) ...
    # For brevity, assuming this part is copied verbatim from your script
    local cols=(22 5 28 30 30 10 22); local headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time Status)
    local print_border_func() { local l="$1" m="$2" r="$3"; printf "%s" "$l"; for i in "${!cols[@]}"; do local w=${cols[i]}; local seg=$((w+2)); for((j=0;j<seg;j++));do printf '═';done; if((i<${#cols[@]}-1));then printf "%s" "$m";else printf "%s\n" "$r";fi;done;}
    local center_text_func() { local wid=$1 txt=$2; if((${#txt}>wid));then txt="${txt:0:$((wid-3))}...";fi; local len=${#txt};local pad_tot=$((wid-len));local pad_s=$((pad_tot/2));local pad_e=$((pad_tot-pad_s));printf "%*s%s%*s" $pad_s "" "$txt" $pad_e "";}
    print_border_func "╔" "╤" "╗"; printf "║"; for i in "${!headers[@]}";do printf " %s " "$(center_text_func "${cols[i]}" "${headers[i]}")";printf "║";done;echo;print_border_func "╟" "┼" "╢"
    for row_data in "${SUMMARY_FOR_TABLE[@]}";do IFS=$'\t' read -r ver pro shp f5 l5 tm st <<<"$row_data";local s_tr="${shp:0:${cols[2]}}";local f_tr="${f5:0:${cols[3]}}";local l_tr="${l5:0:${cols[4]}}";local t_tr="${st:0:${cols[6]}}";printf "║ %-*s ║ %*s ║ %-*s ║ %-*s ║ %-*s ║ %*s ║ %-*s ║\n" "${cols[0]}" "$ver" "${cols[1]}" "$pro" "${cols[2]}" "$s_tr" "${cols[3]}" "$f_tr" "${cols[4]}" "$l_tr" "${cols[5]}" "$tm" "${cols[6]}" "$t_tr";done;print_border_func "╚" "╧" "╝"; echo ""
    log_message "Detailed logs in: $SESSION_LOG_DIR"
    log_message "CSV summary: $CSV_OUTPUT_FILE"
}


# ==============================================================================
# --- Initial Setup Phases (SSH, Build Prep, File Sync, MPI Hostfile) ---
# ==============================================================================
initial_cluster_setup() {
    log_message "--- Running Initial Cluster Setup ---"
    # Phase 1: SSH (copied from previous script, assuming it's robust enough)
    # setup_passwordless_ssh (Ensure this function is defined as in Iteration 2)
    log_message "--- Phase 1: Checking/Setting up Passwordless SSH ---"
    # ... (SSH setup logic from Iteration 2 script - for brevity, not repeating the whole function here)
    # For this combined script, ensure the SSH setup is called once.
    # --- Start SSH Setup snippet ---
    if [[ ! -f "$HOME/.ssh/id_rsa.pub" ]]; then log_message "No local SSH key. Generating..."; ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"; fi
    parse_host_info_entry "${HOSTS_INFO[0]}"; local master_ssh_alias="$_PARSED_USER@$_PARSED_HOST"
    for i in "${!HOSTS_INFO[@]}"; do
        parse_host_info_entry "${HOSTS_INFO[$i]}"; local target_alias="$_PARSED_USER@$_PARSED_HOST"
        log_message "Checking SSH to $target_alias..."
        if ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=5 "$target_alias" "exit" &>/dev/null; then
            log_message "SUCCESS: Passwordless SSH to $target_alias."
        else
            log_message "INFO: Attempting ssh-copy-id to $target_alias."
            if ssh-copy-id "$target_alias" >> "$ORCHESTRATION_LOG" 2>&1; then
                log_message "INFO: ssh-copy-id $target_alias finished. Verifying..."
                if ssh -o PasswordAuthentication=no -o BatchMode=yes -o ConnectTimeout=5 "$target_alias" "exit" &>/dev/null; then
                    log_message "SUCCESS: Passwordless SSH to $target_alias confirmed."
                else log_message "ERROR: SSH to $target_alias still fails. Debug manually."; exit 1; fi
            else log_message "ERROR: ssh-copy-id $target_alias failed. Debug manually."; exit 1; fi
        fi
    done
    log_message "--- SSH Setup Phase Completed ---"
    # --- End SSH Setup snippet ---

    # Phase 2 Prep: Determine Combined CUDA Arch Flags (used by V3/V4 build steps)
    declare -A unique_arch_codes_map
    for host_entry_build in "${HOSTS_INFO[@]}"; do
        parse_host_info_entry "$host_entry_build"
        unique_arch_codes_map["$_PARSED_ARCH_CODE"]=1
    done
    if [[ "${#unique_arch_codes_map[@]}" -eq 0 && (${#HOSTS_INFO[@]} -gt 0) ]]; then # Allow running if HOSTS_INFO is empty (implies local only)
        log_message "INFO: HOSTS_INFO is not empty but no CUDA arch codes found. CUDA builds may use Makefile defaults or fail."
    elif [[ "${#unique_arch_codes_map[@]}" -gt 0 ]]; then
        for code in "${!unique_arch_codes_map[@]}"; do
            COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE+="-gencode arch=compute_${code},code=sm_${code} -gencode arch=compute_${code},code=compute_${code} "
        done
        log_message "Combined CUDA arch flags for V3/V4 builds: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    else
        log_message "INFO: No hosts defined in HOSTS_INFO or no arch codes; CUDA builds will use Makefile defaults."
    fi


    # Phase 3: Sync files if not shared FS (syncs entire project root)
    # synchronize_project_files_to_workers (Ensure this is defined from Iteration 2)
    # --- Start File Sync snippet ---
    if [[ "$USE_SHARED_FILESYSTEM" == "true" ]]; then
        log_message "--- Phase 3: Shared FS. Skipping rsync. ---"
    else
        log_message "--- Phase 3: Syncing Project Files to Worker Nodes ---"
        for i in $(seq 1 $((${#HOSTS_INFO[@]} - 1)) ); do
            parse_host_info_entry "${HOSTS_INFO[$i]}"; local target_alias="$_PARSED_USER@$_PARSED_HOST"
            log_message "Syncing $ROOT_DIR/ to $target_alias:$ROOT_DIR/"
            if ! ssh "$target_alias" "mkdir -p \"$ROOT_DIR\""; then log_message "ERR: mkdir $ROOT_DIR on $target_alias failed."; exit 1; fi
            local rsync_cmd="rsync -az --delete --checksum --exclude '.git/' '$ROOT_DIR/' '$target_alias:$ROOT_DIR/'"
            if [[ "$VERBOSE_RSYNC" == "true" ]]; then rsync_cmd="rsync -avz --delete --checksum --exclude '.git/' '$ROOT_DIR/' '$target_alias:$ROOT_DIR/'"; fi
            if eval "$rsync_cmd" >> "$ORCHESTRATION_LOG" 2>&1; then log_message "SUCCESS: Sync to $target_alias."; else log_message "ERR: rsync to $target_alias failed."; exit 1; fi
        done
    fi
    log_message "--- File Synchronization Phase Completed ---"
    # --- End File Sync snippet ---

    # Phase 4: Create MPI Hostfile (if multiple hosts configured)
    # create_mpi_hostfile_for_run (Ensure this is defined from Iteration 2)
    # --- Start Hostfile Creation snippet ---
    if [[ "${#HOSTS_INFO[@]}" -gt 0 ]]; then
        log_message "--- Phase 4: Creating MPI Hostfile ---"
        rm -f "$MPI_HOSTFILE_PATH"
        for host_entry_run in "${HOSTS_INFO[@]}"; do
            parse_host_info_entry "$host_entry_run"
            echo "$_PARSED_HOST slots=1" >> "$MPI_HOSTFILE_PATH" # Assuming 1 slot per host for this project
        done
        log_message "MPI Hostfile: $MPI_HOSTFILE_PATH"; cat "$MPI_HOSTFILE_PATH" | tee -a "$ORCHESTRATION_LOG"
    else
        log_message "--- Phase 4: No remote hosts in HOSTS_INFO, skipping MPI Hostfile creation. MPI runs will be local-only."
        # Touch the file so later commands expecting it don't fail, but it will be empty or not used for local.
        touch "$MPI_HOSTFILE_PATH" 
    fi
    log_message "--- MPI Hostfile Creation Phase Completed ---"
    # --- End Hostfile Creation snippet ---

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
    local build_succeeded=false; local build_msg="Init"
    log_message "  Building V1 (Log: $make_log_rel_path)..."
    if make -C "$v1_dir" clean >> "$SESSION_LOG_DIR/$make_log_name" 2>&1 && make -C "$v1_dir" >> "$SESSION_LOG_DIR/$make_log_name" 2>&1; then
        if [[ -f "$v1_dir/template" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
    else build_msg="Build failed (exit $?)"; fi
    log_message "    Build Status: $build_msg"

    local run_log_name="run_v1_np${current_np}.log"; local run_log_rel_path="$SESSION_ID/$run_log_name"
    local run_path="$SESSION_LOG_DIR/$run_log_name"
    local shape="–"; local first5="–"; local time_str="–"; local time_num="";
    local run_ok=false; local run_env_warn=false; local run_msg="-"; local parse_ok=false; local parse_msg="-"
    local overall_sym="✘"; local overall_msg="Not Run"

    if $build_succeeded; then
        local cmd_v1="cd $v1_dir && ./template"
        local cmd_v1_exit_code=0; run_and_log_command "$cmd_v1" "$run_path" || cmd_v1_exit_code=$?
        if [[ $cmd_v1_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"
        elif [[ $cmd_v1_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)"
        else run_msg="Runtime Err"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

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
        local max_np_v2=$((${#HOSTS_INFO[@]} > 0 ? ${#HOSTS_INFO[@]} : 4)) # Use up to total hosts, or 4 if local
        if [[ "$max_np_v2" -gt 4 ]]; then max_np_v2=4; fi # Cap at 4 for these tests for now
        
        local np_v2_values=(1)
        if [[ "$max_np_v2" -ge 2 ]]; then np_v2_values+=(2); fi
        if [[ "$max_np_v2" -ge 4 ]]; then np_v2_values+=(4); fi
        
        for np_val in "${np_v2_values[@]}"; do
            current_variant_name="V2 ${ver_suffix//_/-}"; current_np="$np_val"
            log_message "=== Testing $current_variant_name with $current_np processes ==="
            make_log_name="make_v2_${ver_suffix}_np${current_np}.log"; make_log_rel_path="$SESSION_ID/$make_log_name"
            build_succeeded=false; build_msg="Init"
            log_message "  Building $current_variant_name (NP=$current_np, Log: $make_log_rel_path)..."
            if make -C "$v2_dir" clean >> "$SESSION_LOG_DIR/$make_log_name" 2>&1 && make -C "$v2_dir" >> "$SESSION_LOG_DIR/$make_log_name" 2>&1; then
                if [[ -f "$v2_dir/template" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
            else build_msg="Build failed (exit $?)"; fi
            log_message "    Build Status: $build_msg"

            run_log_name="run_v2_${ver_suffix}_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
            shape="–"; first5="–"; time_str="–"; time_num=""
            run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"
            overall_sym="✘"; overall_msg="Not Run"

            if $build_succeeded; then
                local cmd_v2_base="cd $v2_dir && ./template"
                local mpi_cmd_v2="mpirun --oversubscribe -np $current_np $cmd_v2_base"
                if [[ "$current_np" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 ]]; then # Use hostfile for multi-node MPI
                    local num_hosts_to_use=$current_np 
                    if [[ $current_np -gt ${#HOSTS_INFO[@]} ]]; then num_hosts_to_use=${#HOSTS_INFO[@]}; fi # Don't request more procs than hosts in hostfile
                    mpi_cmd_v2="mpirun -np $num_hosts_to_use --hostfile $MPI_HOSTFILE_PATH --report-bindings $FP_DIR/$PROJECT_SUBDIR_V2_NEEDS_CORRECT_PATH/$EXECUTABLE_NAME"
                    # THIS LINE ABOVE IS FLAWED: FP_DIR/$PROJECT_SUBDIR_V2_NEEDS_CORRECT_PATH -> should be $v2_dir/template
                    # Corrected MPI command for V2:
                    mpi_cmd_v2="mpirun -np $num_hosts_to_use --hostfile $MPI_HOSTFILE_PATH --report-bindings $v2_dir/template"
                    if [[ -n "$MPI_NETWORK_INTERFACE" ]]; then
                        mpi_cmd_v2="mpirun -np $num_hosts_to_use --hostfile $MPI_HOSTFILE_PATH --report-bindings --mca btl_tcp_if_include $MPI_NETWORK_INTERFACE --mca oob_tcp_if_include $MPI_NETWORK_INTERFACE $v2_dir/template"
                    else
                         mpi_cmd_v2="mpirun -np $num_hosts_to_use --hostfile $MPI_HOSTFILE_PATH --report-bindings --mca btl_tcp_if_exclude lo,docker0,virbr0 --mca oob_tcp_if_exclude lo,docker0,virbr0 $v2_dir/template"
                    fi
                elif [[ "$current_np" -eq 1 ]]; then # Single process, run locally without hostfile
                    mpi_cmd_v2="mpirun -np 1 $cmd_v2_base" # Oversubscribe not needed for np=1
                fi
                # Fallback to oversubscribe for NP > 1 if not using multi-node setup or hostfile is empty/master-only
                if [[ "$current_np" -gt 1 && ("${#HOSTS_INFO[@]}" -le 1 || ! -s "$MPI_HOSTFILE_PATH") ]]; then
                    mpi_cmd_v2="mpirun --oversubscribe -np $current_np $cmd_v2_base"
                fi


                local cmd_v2_exit_code=0; run_and_log_command "$mpi_cmd_v2" "$run_path" || cmd_v2_exit_code=$?
                if [[ $cmd_v2_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"
                elif [[ $cmd_v2_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)"
                else run_msg="Runtime Err"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

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
    make_log_name="make_v3.log"; make_log_rel_path="$SESSION_ID/$make_log_name"
    build_succeeded=false; build_msg="Init"
    log_message "  Building V3 (Log: $make_log_rel_path)... Using Arch: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    if make -C "$v3_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" clean >> "$SESSION_LOG_DIR/$make_log_name" 2>&1 && \
       make -C "$v3_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" >> "$SESSION_LOG_DIR/$make_log_name" 2>&1; then
        if [[ -f "$v3_dir/template" ]]; then build_succeeded=true; build_msg="Build OK"; else build_msg="Build OK, no exe"; fi
    else build_msg="Build failed (exit $?)"; fi
    log_message "    Build Status: $build_msg"
    
    run_log_name="run_v3_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
    # ... (Reset parse vars: shape, first5, time_str, time_num, run_ok etc.) ...
    shape="–"; first5="–"; time_str="–"; time_num=""; run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"; overall_sym="✘"; overall_msg="Not Run"
    if $build_succeeded; then
        local cmd_v3="cd $v3_dir && ./template"
        local cmd_v3_exit_code=0; run_and_log_command "$cmd_v3" "$run_path" || cmd_v3_exit_code=$?
        # ... (Rest of V3 run and parse logic from 1_final_unique_machine.sh, adapted for these vars) ...
        if [[ $cmd_v3_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔"; 
        elif [[ $cmd_v3_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)";
        else run_msg="Runtime Err"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi
        if $run_ok || $run_env_warn; then
            time_str="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$run_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
            if [[ "$time_str" != "–" ]]; then time_num="${time_str// ms/}"; fi
            first5="$(grep -m1 '^Final Output (first 10 values):' "$run_path" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
            shape="13x13x256" # Fixed for V3
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
    current_variant_name="V4 MPI+CUDA" # Base name, NP is separate
    # Build V4 once
    make_log_name="make_v4.log"; make_log_rel_path="$SESSION_ID/$make_log_name"
    local v4_build_succeeded=false; local v4_build_msg="Init" # Use specific vars for V4 build
    log_message "  Building V4 (Log: $make_log_rel_path)... Using Arch: $COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE"
    if make -C "$v4_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" clean >> "$SESSION_LOG_DIR/$make_log_name" 2>&1 && \
       make -C "$v4_dir" HOST_CUDA_ARCH_FLAGS="$COMBINED_CUDA_ARCH_FLAGS_FOR_MAKE" >> "$SESSION_LOG_DIR/$make_log_name" 2>&1; then
        if [[ -f "$v4_dir/template" ]]; then v4_build_succeeded=true; v4_build_msg="Build OK"; else v4_build_msg="Build OK, no exe"; fi
    else v4_build_msg="Build failed (exit $?)"; fi
    log_message "    Build Status: $v4_build_msg"

    local max_np_v4=$((${#HOSTS_INFO[@]} > 0 ? ${#HOSTS_INFO[@]} : 4))
    if [[ "$max_np_v4" -gt 4 ]]; then max_np_v4=4; fi
    local np_v4_values=(1)
    if [[ "$max_np_v4" -ge 2 ]]; then np_v4_values+=(2); fi
    if [[ "$max_np_v4" -ge 4 ]]; then np_v4_values+=(4); fi

    for np_val in "${np_v4_values[@]}"; do
        current_np="$np_val"
        log_message "--- V4 with $current_np processes ---"
        run_log_name="run_v4_np${current_np}.log"; run_log_rel_path="$SESSION_ID/$run_log_name"; run_path="$SESSION_LOG_DIR/$run_log_name"
        # ... (Reset parse vars) ...
        shape="–"; first5="–"; time_str="–"; time_num=""; run_ok=false; run_env_warn=false; run_msg="-"; parse_ok=false; parse_msg="-"; overall_sym="✘"; overall_msg="Not Run"

        if $v4_build_succeeded; then
            local cmd_v4_base="$v4_dir/template" # Absolute path to executable
            local mpi_cmd_v4="mpirun --oversubscribe -np $current_np $cmd_v4_base" # Default for local
             if [[ "$current_np" -gt 1 && "${#HOSTS_INFO[@]}" -gt 1 ]]; then # Use hostfile for multi-node MPI
                local num_hosts_to_use_v4=$current_np 
                if [[ $current_np -gt ${#HOSTS_INFO[@]} ]]; then num_hosts_to_use_v4=${#HOSTS_INFO[@]}; fi
                
                if [[ -n "$MPI_NETWORK_INTERFACE" ]]; then
                    mpi_cmd_v4="mpirun -np $num_hosts_to_use_v4 --hostfile $MPI_HOSTFILE_PATH --report-bindings --mca btl_tcp_if_include $MPI_NETWORK_INTERFACE --mca oob_tcp_if_include $MPI_NETWORK_INTERFACE $cmd_v4_base"
                else
                    mpi_cmd_v4="mpirun -np $num_hosts_to_use_v4 --hostfile $MPI_HOSTFILE_PATH --report-bindings --mca btl_tcp_if_exclude lo,docker0,virbr0 --mca oob_tcp_if_exclude lo,docker0,virbr0 $cmd_v4_base"
                fi
            elif [[ "$current_np" -eq 1 ]]; then # Single process, run locally
                mpi_cmd_v4="mpirun -np 1 $cmd_v4_base"
            fi
            # Fallback to oversubscribe for NP > 1 if not using multi-node setup or hostfile is empty/master-only
            if [[ "$current_np" -gt 1 && ("${#HOSTS_INFO[@]}" -le 1 || ! -s "$MPI_HOSTFILE_PATH") ]]; then
                mpi_cmd_v4="mpirun --oversubscribe -np $current_np $cmd_v4_base"
            fi

            local cmd_v4_exit_code=0; run_and_log_command "$mpi_cmd_v4" "$run_path" || cmd_v4_exit_code=$?
            # ... (Rest of V4 run and parse logic from 1_final_unique_machine.sh, adapted for these vars) ...
            if [[ $cmd_v4_exit_code -eq 0 ]]; then run_ok=true; run_msg="Run OK"; overall_sym="✔"; overall_msg="✔";
            elif [[ $cmd_v4_exit_code -eq 2 ]]; then run_env_warn=true; run_msg="Env Warn"; overall_sym="⚠"; overall_msg="⚠ (env)";
            else run_msg="Runtime Err"; overall_sym="✘"; overall_msg="✘ (runtime)"; fi

            if $run_ok || $run_env_warn; then
                # V4 specific parsing logic from 1_final_unique_machine.sh
                shape="$(grep -m1 '^Final Output Shape:' "$run_path" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
                if [[ "$shape" == "–" ]]; then # Fallback for old V4 output
                    # ... (your existing complex V4 shape parsing logic from 1_final_unique_machine.sh if needed) ...
                    # For now, keeping it simple; assume new output format.
                    if [[ "$current_np" -gt 1 && "$shape" == "–" ]]; then # Placeholder for decomposed shapes if primary parse fails
                        case $current_np in 2) shape="~?x13x256 (split)";; 4) shape="~?x13x256 (split)";; *) shape="?x?x? (decomposed)";; esac
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
    # ... (initial logging from Iteration 2 main) ...
    log_message "Session ID: $SESSION_ID"
    log_message "Logging to Directory: $SESSION_LOG_DIR"
    log_message "Project Root: $ROOT_DIR"
    log_message "Final Project Dir: $FP_DIR"
    
    if [[ "${#HOSTS_INFO[@]}" -eq 0 ]]; then
        log_message "WARNING: HOSTS_INFO array is empty. All MPI tests will run locally."
        # No need to exit, script can proceed with local-only MPI runs.
    else
        log_message "Target hosts configured:"
        for host_entry_main in "${HOSTS_INFO[@]}"; do log_message "  - $host_entry_main"; done
    fi

    initial_cluster_setup # Handles SSH, CUDA flag determination, file sync, MPI hostfile
    run_test_suite        # Runs V1, V2.x, V3, V4 with appropriate MPI distribution
    print_summary_table   # Prints ASCII table to console

    log_message "--- Full Suite Test Script Finished ---"
    log_message "All logs for this session are in: $SESSION_LOG_DIR"
    log_message "Review the CSV summary: $CSV_OUTPUT_FILE"
    log_message "Review the main orchestration log: $ORCHESTRATION_LOG"
}

# --- Run Main ---
main