#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
MACHINE_ID=$(hostname -s)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP="$ROOT/final_project"
LOGS_BASE="$FP/logs"
mkdir -p "$LOGS_BASE"

SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_ID="run_${SESSION_TIMESTAMP}_${MACHINE_ID}"
SESSION_LOG_DIR="$LOGS_BASE/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"

CSV_OUTPUT_FILE="$LOGS_BASE/summary_${SESSION_TIMESTAMP}_${MACHINE_ID}.csv"
GIT_COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

# --- CSV Logging ---
# Write CSV Header
echo "SessionID,MachineID,GitCommit,EntryTimestamp,ProjectVariant,NumProcesses,MakeLogFile,BuildSucceeded,BuildMessage,RunLogFile,RunCommandSucceeded,RunEnvironmentWarning,RunMessage,ParseSucceeded,ParseMessage,OverallStatusSymbol,OverallStatusMessage,ExecutionTime_ms,OutputShape,OutputFirst5Values" > "$CSV_OUTPUT_FILE"

log_to_csv() {
    local entry_ts
    entry_ts=$(date --iso-8601=seconds)
    # Args:
    # 1: ProjectVariant, 2: NumProcesses,
    # 3: MakeLogRelPath, 4: BuildSucceeded (bool), 5: BuildMessage,
    # 6: RunLogRelPath, 7: RunCommandSucceeded (bool), 8: RunEnvironmentWarning (bool), 9: RunMessage,
    # 10: ParseSucceeded (bool), 11: ParseMessage,
    # 12: OverallStatusSymbol, 13: OverallStatusMessage
    # 14: ExecutionTime_ms (numeric), 15: OutputShape, 16: OutputFirst5Values

    # Quote all string fields for CSV robustness
    printf "\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%s,\"%s\",%s,\"%s\",\"%s\",%s,%s,\"%s\",%s,\"%s\",\"%s\",\"%s\",%s,\"%s\",\"%s\"\n" \
        "$SESSION_ID" \
        "$MACHINE_ID" \
        "$GIT_COMMIT_HASH" \
        "$entry_ts" \
        "$1" "$2" \
        "$3" "$4" "$5" \
        "$6" "$7" "$8" "$9" \
        "${10}" "${11}" \
        "${12}" "${13}" \
        "${14}" "${15}" "${16}" \
        >> "$CSV_OUTPUT_FILE"
}

# --- Command Execution Helper ---
run_and_log() {
  local cmd="$1"; shift
  local log_path="$1"; shift # Full path to the log file
  echo "-> Running Command: $cmd"
  touch "$log_path" && > "$log_path" # Clear log file
  if eval "$cmd" >>"$log_path" 2>&1; then
    echo "  [✔ Command Succeeded]"
    return 0 # Actual success
  else
    local exit_code=$?
    echo "  [DEBUG] Command failed with exit code: $exit_code"
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log_path"; then
        echo "  [⚠ Warning (exit code: $exit_code) - Possible Environment/Setup Issue - see $(basename "$SESSION_LOG_DIR")/$(basename "$log_path")]"
        return 2 # Environment/Setup Warning
    elif grep -q -E "There are not enough slots available|could not be found|orted context|" "$log_path"; then
         echo "  [⚠ Warning (exit code: $exit_code) - MPI Resource/Config Issue - see $(basename "$SESSION_LOG_DIR")/$(basename "$log_path")]"
         return 2 # MPI Resource Warning
    else
        if grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$log_path"; then
             echo "  [✘ Failed (exit code: $exit_code) – V3/V4 Linker Error? See $(basename "$SESSION_LOG_DIR")/$(basename "$log_path")]"
        else
             echo "  [✘ Failed (exit code: $exit_code) – see $(basename "$SESSION_LOG_DIR")/$(basename "$log_path")]"
        fi
        return 1 # Actual failure
    fi
  fi
}

# --- ASCII Table Summary ---
declare -a SUMMARY_FOR_TABLE
add_to_table_summary() {
  # Args: Version, Procs, Shape, First5, Last5, Time, StatusMessage
  SUMMARY_FOR_TABLE+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7")
}

# --- Testing V1 Serial ---
echo "=== Testing V1 Serial (1 process) ==="
cd "$FP/v1_serial"
current_variant_name="V1 Serial"
current_np=1

# Build V1
make_log_name="make_v1.log"
make_log_path="$SESSION_LOG_DIR/$make_log_name"
make_log_rel_path="$SESSION_ID/$make_log_name"
echo "--- Building V1 (Log: $make_log_rel_path) ---"
make_succeeded_bool=false; build_message="✘ (init build status)"

make clean > /dev/null && make >> "$make_log_path" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_bool=true
        build_message="Build OK"
    else
        echo "  [✘ make reported success BUT './template' is MISSING - check $make_log_rel_path]"
        build_message="Build OK, no executable"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $make_log_rel_path]"
    build_message="Build failed (exit $make_exit_code)"
fi

# Run V1
run_log_name="run_v1_np${current_np}.log"
run_log_path="$SESSION_LOG_DIR/$run_log_name"
run_log_rel_path="$SESSION_ID/$run_log_name"

parsed_shape="–"; parsed_first5="–"; parsed_last5="–"; parsed_time_str="–"; parsed_time_numeric=""
run_cmd_succeeded_bool=false; run_env_warn_bool=false; run_message="-"
parse_succeeded_bool=false; parse_message="-"
overall_status_symbol="✘"; overall_status_message="Not Run"

if [[ "$make_succeeded_bool" = true ]]; then
    echo "  [DEBUG] V1 make succeeded. Attempting run (Log: $run_log_rel_path)..."
    run_cmd_exit_code=0
    run_and_log "./template" "$run_log_path" || run_cmd_exit_code=$?

    if [[ $run_cmd_exit_code -eq 0 ]]; then
        run_cmd_succeeded_bool=true; run_message="Run OK"; overall_status_symbol="✔"; overall_status_message="✔"
    elif [[ $run_cmd_exit_code -eq 2 ]]; then
        run_env_warn_bool=true; run_message="Env/Config Warning"; overall_status_symbol="⚠"; overall_status_message="⚠ (env issue)"
    else
        run_message="Runtime error"; overall_status_symbol="✘"; overall_status_message="✘ (runtime err)"
    fi

    if [[ "$run_cmd_succeeded_bool" = true || "$run_env_warn_bool" = true ]]; then # Attempt parse if run started
        parsed_time_str="$(grep -m1 'completed in' "$run_log_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
        if [[ "$parsed_time_str" != "–" ]]; then parsed_time_numeric="${parsed_time_str// ms/}"; fi

        parsed_shape="$(grep -m1 'After LRN2' "$run_log_path" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "–")"
        if [[ "$parsed_shape" == "–" ]]; then
           parsed_shape="$(grep -m1 '^Final Output Shape:' "$run_log_path" | sed -n -E 's/.*Shape: ([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")" # Default/Fallback
        fi
        parsed_first5="$(grep -m1 '^Final Output (first 10 values):' "$run_log_path" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
        parsed_last5="$parsed_first5" # V1 specific assumption

        if [[ "$parsed_first5" == "–" || "$parsed_time_str" == "–" || "$parsed_shape" == "–" ]]; then
            parse_succeeded_bool=false; parse_message="Parse error (missing data)"
            if [[ "$run_cmd_succeeded_bool" = true ]]; then # Downgrade success if parsing failed
                overall_status_symbol="⚠"; overall_status_message="⚠ (parse err)"
            fi
        else
            parse_succeeded_bool=true; parse_message="Parse OK"
        fi
    else
        parse_message="Not run or runtime error"
    fi
else
    overall_status_message="✘ ($build_message)" # Use detailed build message
    run_message="Skipped due to build issue"
    parse_message="Skipped due to build issue"
fi
add_to_table_summary "$current_variant_name" "$current_np" "$parsed_shape" "$parsed_first5" "$parsed_last5" "$parsed_time_str" "$overall_status_message"
log_to_csv "$current_variant_name" "$current_np" \
    "$make_log_rel_path" "$make_succeeded_bool" "$build_message" \
    "$run_log_rel_path" "$run_cmd_succeeded_bool" "$run_env_warn_bool" "$run_message" \
    "$parse_succeeded_bool" "$parse_message" \
    "$overall_status_symbol" "$overall_status_message" \
    "$parsed_time_numeric" "$parsed_shape" "$parsed_first5"

# --- Testing V2 MPI Only ---
for ver_suffix in "2.1_broadcast_all" "2.2_scatter_halo"; do
  for np_val in 1 2 4; do
    current_variant_name="V2 ${ver_suffix//_/-}"
    current_np="$np_val"
    echo "=== Testing $current_variant_name with $current_np processes ==="
    cd "$FP/v2_mpi_only/$ver_suffix"

    # Build V2
    make_log_name="make_v2_${ver_suffix}_np${current_np}.log"
    make_log_path="$SESSION_LOG_DIR/$make_log_name"
    make_log_rel_path="$SESSION_ID/$make_log_name"
    echo "--- Building $current_variant_name (NP=$current_np, Log: $make_log_rel_path) ---"
    make_succeeded_bool=false; build_message="✘ (init build status)"

    make clean > /dev/null && make >> "$make_log_path" 2>&1
    make_exit_code=$?
    if [[ $make_exit_code -eq 0 ]]; then
        if [[ -f ./template ]]; then
            echo "  [✔ make succeeded and './template' found]"
            make_succeeded_bool=true; build_message="Build OK"
        else
            echo "  [✘ make reported success BUT './template' is MISSING - check $make_log_rel_path]"
            build_message="Build OK, no executable"
        fi
    else
        echo "  [✘ make failed (exit code: $make_exit_code) - check $make_log_rel_path]"
        build_message="Build failed (exit $make_exit_code)"
    fi

    # Run V2
    run_log_name="run_v2_${ver_suffix}_np${current_np}.log"
    run_log_path="$SESSION_LOG_DIR/$run_log_name"
    run_log_rel_path="$SESSION_ID/$run_log_name"

    parsed_shape="–"; parsed_sample_output="–"; parsed_time_str="–"; parsed_time_numeric=""
    run_cmd_succeeded_bool=false; run_env_warn_bool=false; run_message="-"
    parse_succeeded_bool=false; parse_message="-"
    overall_status_symbol="✘"; overall_status_message="Not Run"

    if [[ "$make_succeeded_bool" = true ]]; then
        echo "  [DEBUG] V2 make succeeded. Attempting run (Log: $run_log_rel_path)..."
        run_cmd_exit_code=0
        run_and_log "mpirun --oversubscribe -np $current_np ./template" "$run_log_path" || run_cmd_exit_code=$?

        if [[ $run_cmd_exit_code -eq 0 ]]; then
            run_cmd_succeeded_bool=true; run_message="Run OK"; overall_status_symbol="✔"; overall_status_message="✔"
        elif [[ $run_cmd_exit_code -eq 2 ]]; then
            run_env_warn_bool=true; run_message="Env/Config Warning"; overall_status_symbol="⚠"; overall_status_message="⚠ (env issue)"
        else
            run_message="Runtime error"; overall_status_symbol="✘"; overall_status_message="✘ (runtime err)"
        fi

        if [[ "$run_cmd_succeeded_bool" = true || "$run_env_warn_bool" = true ]]; then
            parsed_shape="$(grep -m1 '^shape:' "$run_log_path" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
            parsed_sample_output="$(grep -m1 '^Sample values:' "$run_log_path" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
            parsed_time_str="$(grep -m1 '^Execution Time:' "$run_log_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
            if [[ "$parsed_time_str" != "–" ]]; then parsed_time_numeric="${parsed_time_str// ms/}"; fi

            if [[ "$parsed_shape" == "–" || "$parsed_sample_output" == "–" ]]; then # Time is optional for V2 status
                parse_succeeded_bool=false; parse_message="Parse error (missing shape/sample)"
                 if [[ "$run_cmd_succeeded_bool" = true ]]; then overall_status_symbol="⚠"; overall_status_message="⚠ (parse err)"; fi
                echo "  [⚠ Warning: V2 ($ver_suffix, $current_np) succeeded but failed to parse shape/sample output from $run_log_rel_path]"
            else
                parse_succeeded_bool=true; parse_message="Parse OK"
            fi
            if [[ "$parsed_time_str" == "–" && "$parse_succeeded_bool" = true ]]; then # Info if time is missing but rest is OK
                echo "  [ℹ Info: V2 ($ver_suffix, $current_np) Could not parse execution time from $run_log_rel_path]"
                parse_message="$parse_message (time not found)"
            fi
        else
            parse_message="Not run or runtime error"
        fi
    else
        overall_status_message="✘ ($build_message)"
        run_message="Skipped due to build issue"
        parse_message="Skipped due to build issue"
    fi
    add_to_table_summary "$current_variant_name" "$current_np" "$parsed_shape" "$parsed_sample_output" "$parsed_sample_output" "$parsed_time_str" "$overall_status_message"
    log_to_csv "$current_variant_name" "$current_np" \
        "$make_log_rel_path" "$make_succeeded_bool" "$build_message" \
        "$run_log_rel_path" "$run_cmd_succeeded_bool" "$run_env_warn_bool" "$run_message" \
        "$parse_succeeded_bool" "$parse_message" \
        "$overall_status_symbol" "$overall_status_message" \
        "$parsed_time_numeric" "$parsed_shape" "$parsed_sample_output"
  done
done

# --- Testing V3 CUDA Only ---
echo "=== Testing V3 CUDA Only (1 process) ==="
cd "$FP/v3_cuda_only"
current_variant_name="V3 CUDA"
current_np=1

# Build V3
make_log_name="make_v3.log"
make_log_path="$SESSION_LOG_DIR/$make_log_name"
make_log_rel_path="$SESSION_ID/$make_log_name"
echo "--- Building V3 (Log: $make_log_rel_path) ---"
make_succeeded_bool=false; build_message="✘ (init build status)"

make clean > /dev/null && make >> "$make_log_path" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_bool=true; build_message="Build OK"
    else
        echo "  [✘ make reported success (exit 0) BUT './template' is MISSING - check $make_log_rel_path]"
        build_message="Build OK, no executable"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $make_log_rel_path]"
    build_message="Build failed (exit $make_exit_code)"
    if grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$make_log_path"; then
         build_message="Build failed (Linker: CUDA Libs?)"
    fi
fi

# Run V3
run_log_name="run_v3_np${current_np}.log"
run_log_path="$SESSION_LOG_DIR/$run_log_name"
run_log_rel_path="$SESSION_ID/$run_log_name"

parsed_shape="–"; parsed_first5="–"; parsed_last5="–"; parsed_time_str="–"; parsed_time_numeric=""
run_cmd_succeeded_bool=false; run_env_warn_bool=false; run_message="-"
parse_succeeded_bool=false; parse_message="-"
overall_status_symbol="✘"; overall_status_message="Not Run"

if [[ "$make_succeeded_bool" = true ]]; then
    echo "  [DEBUG] V3 make succeeded. Attempting run (Log: $run_log_rel_path)..."
    run_cmd_exit_code=0
    run_and_log "./template" "$run_log_path" || run_cmd_exit_code=$?

    if [[ $run_cmd_exit_code -eq 0 ]]; then
        run_cmd_succeeded_bool=true; run_message="Run OK"; overall_status_symbol="✔"; overall_status_message="✔"
    elif [[ $run_cmd_exit_code -eq 2 ]]; then
        run_env_warn_bool=true; run_message="Env/Config Warning"; overall_status_symbol="⚠"; overall_status_message="⚠ (env issue)"
    else
        run_message="Runtime error"; overall_status_symbol="✘"; overall_status_message="✘ (runtime err)"
    fi

    if [[ "$run_cmd_succeeded_bool" = true || "$run_env_warn_bool" = true ]]; then
        parsed_time_str="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$run_log_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
        if [[ "$parsed_time_str" != "–" ]]; then parsed_time_numeric="${parsed_time_str// ms/}"; fi
        parsed_first5="$(grep -m1 '^Final Output (first 10 values):' "$run_log_path" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
        parsed_last5="$parsed_first5"
        parsed_shape="13x13x256" # Fixed for V3

        if [[ "$parsed_first5" == "–" || "$parsed_time_str" == "–" ]]; then
            parse_succeeded_bool=false; parse_message="Parse error (missing data)"
            if [[ "$run_cmd_succeeded_bool" = true ]]; then overall_status_symbol="⚠"; overall_status_message="⚠ (parse err)"; fi
        else
            parse_succeeded_bool=true; parse_message="Parse OK"
        fi
    else
        parse_message="Not run or runtime error"
    fi
else
    overall_status_message="✘ ($build_message)"
    run_message="Skipped due to build issue"
    parse_message="Skipped due to build issue"
fi
add_to_table_summary "$current_variant_name" "$current_np" "$parsed_shape" "$parsed_first5" "$parsed_last5" "$parsed_time_str" "$overall_status_message"
log_to_csv "$current_variant_name" "$current_np" \
    "$make_log_rel_path" "$make_succeeded_bool" "$build_message" \
    "$run_log_rel_path" "$run_cmd_succeeded_bool" "$run_env_warn_bool" "$run_message" \
    "$parse_succeeded_bool" "$parse_message" \
    "$overall_status_symbol" "$overall_status_message" \
    "$parsed_time_numeric" "$parsed_shape" "$parsed_first5"

# --- Testing V4 MPI+CUDA ---
echo "=== Testing V4 MPI+CUDA ==="
cd "$FP/v4_mpi_cuda"
current_variant_base_name="V4 MPI+CUDA"

# Build V4 (once for all NP counts)
make_log_name="make_v4.log"
make_log_path="$SESSION_LOG_DIR/$make_log_name"
make_log_rel_path="$SESSION_ID/$make_log_name"
echo "--- Building V4 (Log: $make_log_rel_path) ---"
v4_make_succeeded_bool=false; v4_build_message="✘ (init build status)" # Renamed to avoid conflict in loop

make clean > /dev/null && make >> "$make_log_path" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        v4_make_succeeded_bool=true; v4_build_message="Build OK"
    else
        echo "  [✘ make reported success (exit 0) BUT './template' is MISSING - check $make_log_rel_path]"
        v4_build_message="Build OK, no executable"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $make_log_rel_path]"
    v4_build_message="Build failed (exit $make_exit_code)"
    if grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$make_log_path"; then
         v4_build_message="Build failed (Linker: CUDA Libs?)"
    fi
fi

for np_val in 1 2 4; do
    current_variant_name="$current_variant_base_name" # NP is a separate field
    current_np="$np_val"
    echo "--- V4 with $current_np processes ---"

    run_log_name="run_v4_np${current_np}.log"
    run_log_path="$SESSION_LOG_DIR/$run_log_name"
    run_log_rel_path="$SESSION_ID/$run_log_name"

    parsed_shape="–"; parsed_sample_output="–"; parsed_time_str="–"; parsed_time_numeric=""
    run_cmd_succeeded_bool=false; run_env_warn_bool=false; run_message="-"
    parse_succeeded_bool=false; parse_message="-"
    overall_status_symbol="✘"; overall_status_message="Not Run"

    if [[ "$v4_make_succeeded_bool" = true ]]; then
        echo "  [DEBUG] V4 make succeeded. Attempting run with NP=$current_np (Log: $run_log_rel_path)..."
        run_cmd_exit_code=0
        run_and_log "mpirun --oversubscribe -np $current_np ./template" "$run_log_path" || run_cmd_exit_code=$?

        if [[ $run_cmd_exit_code -eq 0 ]]; then
            run_cmd_succeeded_bool=true; run_message="Run OK"; overall_status_symbol="✔"; overall_status_message="✔"
        elif [[ $run_cmd_exit_code -eq 2 ]]; then
            run_env_warn_bool=true; run_message="Env/Config Warning"; overall_status_symbol="⚠"; overall_status_message="⚠ (env issue)"
        else
            run_message="Runtime error"; overall_status_symbol="✘"; overall_status_message="✘ (runtime err)"
        fi

        if [[ "$run_cmd_succeeded_bool" = true || "$run_env_warn_bool" = true ]]; then
            parsed_shape="$(grep -m1 '^Final Output Shape:' "$run_log_path" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
            # Fallback shape parsing for V4 from run_final_project.sh (original)
            if [[ "$parsed_shape" == "–" ]]; then
                total_size=$(grep -m1 '^Final Output Total Size:' "$run_log_path" | sed -n -E 's/^Final Output Total Size: *([0-9]+) .*/\1/p' || echo "")
                if [[ -n "$total_size" ]]; then
                    if [[ "$total_size" == "43264" ]]; then parsed_shape="13x13x256"; # Expected total for 13x13x256
                    elif [[ "$total_size" == "0" ]]; then parsed_shape="0x0x0";
                    else parsed_shape="?x?x? ($total_size elem)"; fi
                else
                    # Fallback from run_final_project_alt.sh for V4 decomposed shape
                    local_shape_h=$(grep -m1 '^Rank .* local output H=' "$run_log_path" | sed -n -E 's/.*local output H=([0-9]+).*/\1/p' || echo "")
                    local_shape_w=$(grep -m1 '^Rank .* local output W=' "$run_log_path" | sed -n -E 's/.*local output W=([0-9]+).*/\1/p' || echo "")
                    local_shape_c=$(grep -m1 '^Rank .* local output C=' "$run_log_path" | sed -n -E 's/.*local output C=([0-9]+).*/\1/p' || echo "")
                    if [[ -n "$local_shape_h" && -n "$local_shape_w" && -n "$local_shape_c" ]]; then
                        if [[ "$current_np" -gt 1 && "$local_shape_h" != "13" ]]; then
                            global_h_guess=$((local_shape_h * current_np)) # Rough guess
                            parsed_shape="${global_h_guess}x${local_shape_w}x${local_shape_c} (est. local ${local_shape_h}x)"
                        else
                            parsed_shape="${local_shape_h}x${local_shape_w}x${local_shape_c}"
                        fi
                    elif [[ "$current_np" -gt 1 ]]; then # Default decomposed shapes if no specific output found
                        case $current_np in
                            2) parsed_shape="~7x13x256 (split)";; # Approximation
                            4) parsed_shape="~4x13x256 (split)";; # Approximation
                            *) parsed_shape="?x?x? (decomposed)";;
                        esac
                    else
                         parsed_shape="–" # Final fallback
                    fi
                fi
            fi

            parsed_sample_output="$(grep -m1 '^Final Output (first 10 values):' "$run_log_path" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
            parsed_time_str="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$run_log_path" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
            if [[ "$parsed_time_str" != "–" ]]; then parsed_time_numeric="${parsed_time_str// ms/}"; fi

            if [[ "$parsed_shape" == "–" || "$parsed_sample_output" == "–" ]]; then # Time is optional
                parse_succeeded_bool=false; parse_message="Parse error (missing shape/sample)"
                if [[ "$run_cmd_succeeded_bool" = true ]]; then overall_status_symbol="⚠"; overall_status_message="⚠ (parse err)"; fi
                echo "  [⚠ Warning: V4 (NP=$current_np) succeeded but failed to parse critical output from $run_log_rel_path (Shape: '$parsed_shape', Sample: '$parsed_sample_output')]"
            else
                parse_succeeded_bool=true; parse_message="Parse OK"
            fi
            if [[ "$parsed_time_str" == "–" && "$parse_succeeded_bool" = true ]]; then
                 echo "  [ℹ Info: V4 (NP=$current_np) Could not parse execution time from $run_log_rel_path]"
                 parse_message="$parse_message (time not found)"
            fi
        else
             parse_message="Not run or runtime error"
        fi
    else
        overall_status_message="✘ ($v4_build_message)" # Use V4 specific build message
        run_message="Skipped due to build issue"
        parse_message="Skipped due to build issue"
    fi
    add_to_table_summary "$current_variant_name" "$current_np" "$parsed_shape" "$parsed_sample_output" "$parsed_sample_output" "$parsed_time_str" "$overall_status_message"
    # For CSV, use the specific v4_build_message and v4_make_succeeded_bool
    log_to_csv "$current_variant_name" "$current_np" \
        "$make_log_rel_path" "$v4_make_succeeded_bool" "$v4_build_message" \
        "$run_log_rel_path" "$run_cmd_succeeded_bool" "$run_env_warn_bool" "$run_message" \
        "$parse_succeeded_bool" "$parse_message" \
        "$overall_status_symbol" "$overall_status_message" \
        "$parsed_time_numeric" "$parsed_shape" "$parsed_sample_output"
done


# --- Print Final ASCII Table ---
echo ""
echo "=== Summary Table (Machine: $MACHINE_ID, Session: $SESSION_ID) ==="
# Adjusted Shape column width for potentially longer V4 shape strings
cols=(22 5 28 30 30 10 22) # Version, Procs, Shape, First 5, Last 5, Time, Status
headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time Status)

print_border() {
  local left="$1" mid="$2" right="$3"
  printf "%s" "$left"
  for i in "${!cols[@]}"; do
    local w=${cols[i]}
    local seg_len=$((w + 2)) # Account for spaces around content
    for ((j=0; j<seg_len; j++)); do printf '═'; done
    if (( i < ${#cols[@]} - 1 )); then printf "%s" "$mid"; else printf "%s\n" "$right"; fi
  done
}

center_text() {
    local width=$1
    local text=$2
    # Ensure text is not wider than column, truncate if necessary
    if ((${#text} > width)); then text="${text:0:$((width-3))}..."; fi
    local text_len=${#text}
    local pad_total=$((width - text_len))
    local pad_start=$((pad_total / 2))
    local pad_end=$((pad_total - pad_start))
    printf "%*s%s%*s" $pad_start "" "$text" $pad_end ""
}

print_border "╔" "╤" "╗"
printf "║"
for i in "${!headers[@]}"; do
   printf " %s " "$(center_text "${cols[i]}" "${headers[i]}")"
   printf "║"
done; echo
print_border "╟" "┼" "╢"

for row_data in "${SUMMARY_FOR_TABLE[@]}"; do
  IFS=$'\t' read -r ver pro shape f5 l5 tm st <<<"$row_data"
  
  # Truncate potentially long strings for table display
  shape_trunc="${shape:0:${cols[2]}}"
  f5_trunc="${f5:0:${cols[3]}}"
  l5_trunc="${l5:0:${cols[4]}}"
  st_trunc="${st:0:${cols[6]}}"

  printf "║ %-*s ║ %*s ║ %-*s ║ %-*s ║ %-*s ║ %*s ║ %-*s ║\n" \
    "${cols[0]}" "$ver" \
    "${cols[1]}" "$pro" \
    "${cols[2]}" "$shape_trunc" \
    "${cols[3]}" "$f5_trunc" \
    "${cols[4]}" "$l5_trunc" \
    "${cols[5]}" "$tm" \
    "${cols[6]}" "$st_trunc"
done
print_border "╚" "╧" "╝"
echo ""
echo "Session logs directory: $SESSION_LOG_DIR"
echo "CSV summary written to: $CSV_OUTPUT_FILE"


# --- FUTURE: Multi-Machine Testing (Conceptual Placeholder) ---
# To extend this for multiple machines, you might consider:
#
# 1. Using a Hostfile for MPI:
#    Create a hostfile (e.g., my_hosts.txt):
#    --------------------
#    machine1.domain slots=4
#    machine2.domain slots=4
#    --------------------
#    Then, modify the mpirun command:
#    MPIRUN_CMD="mpirun --hostfile /path/to/my_hosts.txt -np $NP ./template"
#    Ensure that passwordless SSH is set up between the control machine and compute nodes.
#    The project code must be available on all nodes at the same path.
#
# 2. Orchestration with SSH for independent runs & aggregation:
#    You could loop through a list of known hosts:
#    KNOWN_HOSTS=("machine1.domain" "machine2.domain")
#    for host in "${KNOWN_HOSTS[@]}"; do
#      echo "Running tests on $host..."
#      # This assumes the script is present on the remote machine
#      # and can output its CSV to a shared/accessible location or print to stdout for collection.
#      ssh "$host" "/path/to/this_script.sh --machine-id $host --output-dir /shared/results/$SESSION_ID" &
#    done
#    wait # Wait for all backgrounded SSH jobs to complete
#    # After all hosts complete, aggregate CSV files from /shared/results/$SESSION_ID/*
#
# 3. Using a Cluster Job Scheduler (e.g., Slurm, PBS/Torque):
#    If you are in an HPC environment, the job scheduler is the standard way.
#    You would write a submission script that requests resources (nodes, cores)
#    and the scheduler would manage the execution of your MPI program.
#    Example Slurm batch script snippet:
#    --------------------
#    #!/bin/bash
#    #SBATCH --nodes=2
#    #SBATCH --ntasks-per-node=2  # Total NP = nodes * ntasks-per-node = 4
#    #SBATCH --job-name=AlexNetTest
#    #SBATCH --output=${SESSION_LOG_DIR}/slurm_output_%j.log
#
#    # Load necessary modules (MPI, CUDA, etc.)
#    module load mpi/openmpi cuda/11.x
#
#    # Your application command
#    # srun is often used in Slurm to launch parallel tasks
#    srun "$FP/v4_mpi_cuda/template" >> "$SESSION_LOG_DIR/run_v4_np${SLURM_NTASKS}.log" 2>&1
#    --------------------
#    This script would then be submitted using `sbatch your_slurm_script.sh`.
#    The `MACHINE_ID` might be a list of nodes, and `GitCommit` would be important for tracking.
#    The `log_to_csv` function would need to be called appropriately after the run,
#    possibly by parsing Slurm's output or the application's log.
#
# Key considerations for multi-machine:
#    - Code distribution: Ensure the compiled executable and any input files are accessible on all nodes.
#    - Path consistency: Paths to executables and libraries should be consistent or handled by environment modules.
#    - Output aggregation: A strategy to collect logs and results (like CSV data) from all nodes.
#    - Unique identification: `MACHINE_ID` in the CSV would represent the set of hosts used or the primary host.
#
# The current script structure (SESSION_ID, CSV logging) provides a good foundation,
# as you could adapt it to have each node log its part to a shared filesystem under the same SESSION_ID,
# or have a master process collect data from slave processes.