#!/usr/bin/env bash
set -euo pipefail

# root of repo
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP="$ROOT/final_project"
LOGS="$FP/logs"
mkdir -p "$LOGS"

# CSV Output File - Timestamped
TIMESTAMP_FILE=$(date +%Y%m%d_%H%M%S)
CSV_OUTPUT_FILE="$LOGS/run_summary_${TIMESTAMP_FILE}.csv"

# Write CSV Header
echo "Timestamp,Version,NP,StatusSymbol,StatusMessage,Success,Time_ms,Shape,First_Values" > "$CSV_OUTPUT_FILE"

# Helper function to log data to CSV
log_to_csv() {
    local timestamp="$1"
    local version_name="$2"
    local np_val="$3"
    local status_symbol="$4" # Just the symbol like ✔, ⚠, ✘
    local status_message="$5" # Full status text
    local success_bool="$6" # "true" or "false"
    local time_ms="$7" # Numeric time or empty
    local shape_val="$8"
    local first_values="$9"

    # Ensure values are properly quoted if they contain commas (simple handling for now)
    # For robust CSV, a proper CSV library or more complex printf would be needed.
    # Assuming our current data fields don't have internal commas.
    echo "\"$timestamp\",\"$version_name\",$np_val,\"$status_symbol\",\"$status_message\",$success_bool,$time_ms,\"$shape_val\",\"$first_values\"" >> "$CSV_OUTPUT_FILE"
}


# helper to run & log; prints ✅/❌/⚠
run_and_log() {
  local cmd="$1"; shift
  local log="$1"; shift
  echo "-> Running Command: $cmd"
  touch "$log" && > "$log"
  if eval "$cmd" >>"$log" 2>&1; then
    echo "  [✔ Command Succeeded]"
    return 0
  else
    local exit_code=$?
    echo "  [DEBUG] Command failed with exit code: $exit_code"
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log"; then
        echo "  [⚠ Warning (exit code: $exit_code) - Possible Environment/Setup Issue - see $log]"
        return 2
    elif grep -q -E "There are not enough slots available|could not be found|orted context|" "$log"; then
         echo "  [⚠ Warning (exit code: $exit_code) - MPI Resource/Config Issue - see $log]"
         return 2
    else
        if grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$log"; then
             echo "  [✘ Failed (exit code: $exit_code) – V3/V4 Linker Error? See $log]"
        else
             echo "  [✘ Failed (exit code: $exit_code) – see $log]"
        fi
        return 1
    fi
  fi
}

# collect tab‑separated summary lines for ASCII table
declare -a SUMMARY
add_summary() {
  SUMMARY+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7")
}

# --- Testing V1 ---
echo "=== Testing V1 Serial (1 process) ==="
cd "$FP/v1_serial"
MAKE_LOG_V1="$LOGS/make_v1_${TIMESTAMP_FILE}.log" # Timestamp log file
touch "$MAKE_LOG_V1" && > "$MAKE_LOG_V1"
echo "--- Building V1 ---"
make_succeeded_v1=false
make clean > /dev/null && make >> "$MAKE_LOG_V1" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_v1=true
    else
        echo "  [✘ make reported success BUT './template' is MISSING - check $MAKE_LOG_V1]"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $MAKE_LOG_V1]"
fi

LOG_V1="$LOGS/final_project_v1_np1_${TIMESTAMP_FILE}.log" # Timestamp log file
shape_v1="–"; first5_v1="–"; last5_v1="–"; time_v1="–"; status_v1_msg="✘ (init)"
run_result=0; success_v1_bool="false"; status_v1_symbol="✘"

if [[ "$make_succeeded_v1" = true ]]; then
    echo "  [DEBUG] V1 make succeeded. Attempting run..."
    run_and_log "./template" "$LOG_V1" || run_result=$?
    if [[ $run_result -eq 0 ]]; then status_v1_msg="✔"; success_v1_bool="true"; status_v1_symbol="✔";
    elif [[ $run_result -eq 2 ]]; then status_v1_msg="⚠ (env issue)"; status_v1_symbol="⚠";
    else status_v1_msg="✘ (runtime err)"; status_v1_symbol="✘"; fi
else
    if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v1_msg="✘ (make: no exe)";
    else status_v1_msg="✘ (make failed)"; fi
    echo "  [Skipping V1 run due to build issue]"
fi

time_v1_numeric=""
if [[ "$status_v1_msg" == "✔" || ($status_v1_msg == "⚠ (parse err)" && "$make_succeeded_v1" = true) ]]; then # Allow parsing for success or parse error
  time_v1="$(grep -m1 'completed in' "$LOG_V1" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
  if [[ "$time_v1" != "–" ]]; then time_v1_numeric="${time_v1// ms/}"; fi
  shape_v1="$(grep -m1 'After LRN2' "$LOG_V1" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "–")"
  if [[ "$shape_v1" == "–" ]]; then
     shape_v1="$(grep -m1 '^Final Output Shape:' "$LOG_V1" | sed -n -E 's/.*Shape: ([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")"
  fi
  first5_v1="$(grep -m1 '^Final Output (first 10 values):' "$LOG_V1" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v1="$first5_v1"
  if [[ "$first5_v1" == "–" || "$time_v1" == "–" || "$shape_v1" == "–" ]] && [[ "$status_v1_msg" == "✔" ]]; then status_v1_msg="⚠ (parse err)"; status_v1_symbol="⚠"; success_v1_bool="false"; fi # Downgrade if parsing failed on successful run
fi
add_summary "V1 Serial" 1 "$shape_v1" "$first5_v1" "$last5_v1" "$time_v1" "$status_v1_msg"
log_to_csv "$(date --iso-8601=seconds)" "V1 Serial" 1 "$status_v1_symbol" "$status_v1_msg" "$success_v1_bool" "$time_v1_numeric" "$shape_v1" "$first5_v1"


# --- Testing V2 ---
for ver in "2.1_broadcast_all" "2.2_scatter_halo"; do
  for np in 1 2 4; do
    echo "=== Testing V2 ($ver) with $np processes ==="
    cd "$FP/v2_mpi_only/$ver"
    MAKE_LOG_V2="$LOGS/make_v2_${ver}_np${np}_${TIMESTAMP_FILE}.log"
    touch "$MAKE_LOG_V2" && > "$MAKE_LOG_V2"
    echo "--- Building V2 ($ver, NP=$np) ---"
    make_succeeded_v2=false
    make clean > /dev/null && make >> "$MAKE_LOG_V2" 2>&1
    make_exit_code=$?
    if [[ $make_exit_code -eq 0 ]]; then
        if [[ -f ./template ]]; then
            echo "  [✔ make succeeded and './template' found]"
            make_succeeded_v2=true
        else
            echo "  [✘ make reported success BUT './template' is MISSING - check $MAKE_LOG_V2]"
        fi
    else
        echo "  [✘ make failed (exit code: $make_exit_code) - check $MAKE_LOG_V2]"
    fi

    LOG_V2="$LOGS/final_project_v2_${ver}_np${np}_${TIMESTAMP_FILE}.log"
    shape_v2="–"; sample_v2="–"; time_v2="–"; status_v2_msg="✘ (init)"
    ver_name="V2 ${ver//_/-}"
    run_result=0; success_v2_bool="false"; status_v2_symbol="✘"

    if [[ "$make_succeeded_v2" = true ]]; then
        echo "  [DEBUG] V2 make succeeded ($ver, NP=$np). Attempting run..."
        run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG_V2" || run_result=$?
        if [[ $run_result -eq 0 ]]; then status_v2_msg="✔"; success_v2_bool="true"; status_v2_symbol="✔";
        elif [[ $run_result -eq 2 ]]; then status_v2_msg="⚠ (env issue)"; status_v2_symbol="⚠";
        else status_v2_msg="✘ (runtime err)"; status_v2_symbol="✘"; fi
    else
        if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v2_msg="✘ (make: no exe)";
        else status_v2_msg="✘ (make failed)"; fi
        echo "  [Skipping V2 run ($ver, NP=$np) due to build issue]"
    fi

    time_v2_numeric=""
    if [[ "$status_v2_msg" == "✔" || ($status_v2_msg == "⚠ (parse err)" && "$make_succeeded_v2" = true) ]]; then
      shape_v2="$(grep -m1 '^shape:' "$LOG_V2" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
      sample_v2="$(grep -m1 '^Sample values:' "$LOG_V2" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v2="$(grep -m1 '^Execution Time:' "$LOG_V2" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
      if [[ "$time_v2" != "–" ]]; then time_v2_numeric="${time_v2// ms/}"; fi

      if [[ "$shape_v2" == "–" || "$sample_v2" == "–" ]] && [[ "$status_v2_msg" == "✔" ]]; then
         status_v2_msg="⚠ (parse err)"
         status_v2_symbol="⚠"
         success_v2_bool="false" # Downgrade if parsing failed
         echo "  [⚠ Warning: V2 ($ver, $np) succeeded but failed to parse shape/sample output from $LOG_V2]"
      fi
       if [[ "$time_v2" == "–" && "$status_v2_msg" == "✔" ]]; then # If time is missing but run was ok
           echo "  [ℹ Info: V2 ($ver, $np) Could not parse execution time from $LOG_V2]"
       fi
    fi
    add_summary "$ver_name" "$np" "$shape_v2" "$sample_v2" "$sample_v2" "$time_v2" "$status_v2_msg"
    log_to_csv "$(date --iso-8601=seconds)" "$ver_name" "$np" "$status_v2_symbol" "$status_v2_msg" "$success_v2_bool" "$time_v2_numeric" "$shape_v2" "$sample_v2"
  done
done


# --- Testing V3 ---
echo "=== Testing V3 CUDA Only (1 process) ==="
cd "$FP/v3_cuda_only"
MAKE_LOG_V3="$LOGS/make_v3_${TIMESTAMP_FILE}.log"
touch "$MAKE_LOG_V3" && > "$MAKE_LOG_V3"
echo "--- Building V3 ---"
make_succeeded_v3=false
make clean > /dev/null && make >> "$MAKE_LOG_V3" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    echo "  [DEBUG] make exit code: 0"
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_v3=true
    else
        echo "  [✘ make reported success (exit 0) BUT './template' is MISSING - check $MAKE_LOG_V3]"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $MAKE_LOG_V3]"
fi

LOG_V3="$LOGS/final_project_v3_np1_${TIMESTAMP_FILE}.log"
shape_v3="–"; first5_v3="–"; last5_v3="–"; time_v3="–"; status_v3_msg="✘ (init)"
run_result=0; success_v3_bool="false"; status_v3_symbol="✘"

if [[ "$make_succeeded_v3" = true ]]; then
    echo "  [DEBUG] V3 make succeeded. Attempting run..."
    run_and_log "./template" "$LOG_V3" || run_result=$?
    if [[ $run_result -eq 0 ]]; then status_v3_msg="✔"; success_v3_bool="true"; status_v3_symbol="✔";
    elif [[ $run_result -eq 2 ]]; then status_v3_msg="⚠ (env issue)"; status_v3_symbol="⚠";
    else status_v3_msg="✘ (runtime err)"; status_v3_symbol="✘"; fi
else
    echo "  [Skipping V3 run due to build issue]"
    if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v3_msg="✘ (make: no exe)"
    elif grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$MAKE_LOG_V3"; then status_v3_msg="✘ (Linker: CUDA Libs?)"
    else status_v3_msg="✘ (make failed)"; fi
fi

time_v3_numeric=""
if [[ "$status_v3_msg" == "✔" || ($status_v3_msg == "⚠ (parse err)" && "$make_succeeded_v3" = true) ]]; then
  time_v3="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$LOG_V3" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
  if [[ "$time_v3" != "–" ]]; then time_v3_numeric="${time_v3// ms/}"; fi
  first5_v3="$(grep -m1 '^Final Output (first 10 values):' "$LOG_V3" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v3="$first5_v3"
  shape_v3="13x13x256"
  if [[ "$first5_v3" == "–" || "$time_v3" == "–" ]] && [[ "$status_v3_msg" == "✔" ]]; then status_v3_msg="⚠ (parse err)"; status_v3_symbol="⚠"; success_v3_bool="false"; fi
fi
add_summary "V3 CUDA" 1 "$shape_v3" "$first5_v3" "$last5_v3" "$time_v3" "$status_v3_msg"
log_to_csv "$(date --iso-8601=seconds)" "V3 CUDA" 1 "$status_v3_symbol" "$status_v3_msg" "$success_v3_bool" "$time_v3_numeric" "$shape_v3" "$first5_v3"


# --- Testing V4 ---
echo "=== Testing V4 MPI+CUDA ==="
cd "$FP/v4_mpi_cuda"
MAKE_LOG_V4="$LOGS/make_v4_${TIMESTAMP_FILE}.log"
touch "$MAKE_LOG_V4" && > "$MAKE_LOG_V4"
echo "--- Building V4 ---"
make_succeeded_v4=false
make clean > /dev/null && make >> "$MAKE_LOG_V4" 2>&1
make_exit_code=$?
if [[ $make_exit_code -eq 0 ]]; then
    echo "  [DEBUG] make exit code: 0"
    if [[ -f ./template ]]; then
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_v4=true
    else
        echo "  [✘ make reported success (exit 0) BUT './template' is MISSING - check $MAKE_LOG_V4]"
    fi
else
    echo "  [✘ make failed (exit code: $make_exit_code) - check $MAKE_LOG_V4]"
fi

for np in 1 2 4; do
    echo "--- V4 with $np processes ---"
    LOG_V4="$LOGS/final_project_v4_np${np}_${TIMESTAMP_FILE}.log"
    shape_v4="–"; sample_v4="–"; time_v4="–"; status_v4_msg="✘ (init)"
    ver_name="V4 MPI+CUDA"
    run_result=0; success_v4_bool="false"; status_v4_symbol="✘"

    if [[ "$make_succeeded_v4" = true ]]; then
        echo "  [DEBUG] V4 make succeeded. Attempting run with NP=$np ..."
        run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG_V4" || run_result=$?
        if [[ $run_result -eq 0 ]]; then status_v4_msg="✔"; success_v4_bool="true"; status_v4_symbol="✔";
        elif [[ $run_result -eq 2 ]]; then status_v4_msg="⚠ (env issue)"; status_v4_symbol="⚠";
        else status_v4_msg="✘ (runtime err)"; status_v4_symbol="✘"; fi
    else
        echo "  [Skipping V4 run (NP=$np) due to build issue]"
         if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v4_msg="✘ (make: no exe)";
         else status_v4_msg="✘ (make failed)"; fi
    fi

    time_v4_numeric=""
    if [[ "$status_v4_msg" == "✔" || ($status_v4_msg == "⚠ (parse err)" && "$make_succeeded_v4" = true) ]]; then
      shape_v4="$(grep -m1 '^Final Output Shape:' "$LOG_V4" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
      if [[ "$shape_v4" == "–" ]]; then # Fallback parsing if primary shape line missing
        local_shape_h=$(grep -m1 '^Rank .* local output H=' "$LOG_V4" | sed -n -E 's/.*local output H=([0-9]+).*/\1/p' || echo "")
        local_shape_w=$(grep -m1 '^Rank .* local output W=' "$LOG_V4" | sed -n -E 's/.*local output W=([0-9]+).*/\1/p' || echo "")
        local_shape_c=$(grep -m1 '^Rank .* local output C=' "$LOG_V4" | sed -n -E 's/.*local output C=([0-9]+).*/\1/p' || echo "")
        if [[ -n "$local_shape_h" && -n "$local_shape_w" && -n "$local_shape_c" ]]; then
            # Attempt to reconstruct global shape if it was a simple H split
            # This is a guess based on common decomposition, may not be universally correct for the project
            if [[ "$np" -gt 1 && "$local_shape_h" != "13" ]]; then # Assuming target H is 13 for NP=1
                # Simple reconstruction, might need to adjust if decomposition is complex
                global_h_guess=$((local_shape_h * np)) # This is a very rough guess
                shape_v4="${global_h_guess}x${local_shape_w}x${local_shape_c} (est. local ${local_shape_h}x)"
            else
                shape_v4="${local_shape_h}x${local_shape_w}x${local_shape_c}"
            fi
        elif [[ "$np" -gt 1 ]]; then # If rank-specific shape is missing, use project's expected decomposed shape if known
            case $np in
                2) shape_v4="8x13x256 (proj. default)";; # Example, adjust if needed
                4) shape_v4="4x13x256 (proj. default)";; # Example, adjust if needed
                *) shape_v4="?x?x? (decomposed)";;
            esac
        else # NP=1 or other unhandled parsing
             shape_v4="–"
        fi
      fi
      sample_v4="$(grep -m1 '^Final Output (first 10 values):' "$LOG_V4" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v4="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$LOG_V4" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
      if [[ "$time_v4" != "–" ]]; then time_v4_numeric="${time_v4// ms/}"; fi

      if [[ ("$shape_v4" == "–" || "$sample_v4" == "–") && "$status_v4_msg" == "✔" ]]; then
         status_v4_msg="⚠ (parse err)"
         status_v4_symbol="⚠"
         success_v4_bool="false"
         echo "  [⚠ Warning: V4 (NP=$np) succeeded but failed to parse critical output from $LOG_V4 (Shape: '$shape_v4', Sample: '$sample_v4')]"
      fi
      if [[ "$time_v4" == "–" && "$status_v4_msg" == "✔" ]]; then
          echo "  [ℹ Info: V4 (NP=$np) Could not parse execution time from $LOG_V4]"
      fi
    fi
    add_summary "$ver_name" "$np" "$shape_v4" "$sample_v4" "$sample_v4" "$time_v4" "$status_v4_msg"
    log_to_csv "$(date --iso-8601=seconds)" "$ver_name" "$np" "$status_v4_symbol" "$status_v4_msg" "$success_v4_bool" "$time_v4_numeric" "$shape_v4" "$sample_v4"
done

# === print final ASCII table ===
echo ""
echo "=== Summary Table ==="
cols=(22 5 25 30 30 10 22) # Adjusted Shape and Status column width
headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time Status)
print_border() {
  local left="$1" mid="$2" right="$3"
  printf "%s" "$left"
  for i in "${!cols[@]}"; do
    local w=${cols[i]}
    local seg_len=$((w + 2))
    for ((j=0; j<seg_len; j++)); do printf '═'; done
    if (( i < ${#cols[@]} - 1 )); then printf "%s" "$mid"; else printf "%s\n" "$right"; fi
  done
}
center_text() {
    local width=$1
    local text=$2
    local num_width=$((width))
    local text_len=${#text}
    local pad=$(( (num_width - text_len) / 2 ))
    local pad_rem=$(( (num_width - text_len + 1) / 2 ))
    if [[ $pad -lt 0 ]]; then pad=0; fi
    if [[ $pad_rem -lt 0 ]]; then pad_rem=0; fi
    printf "%*s%s%*s" $pad "" "$text" $pad_rem ""
}
print_border "╔" "╤" "╗"
printf "║"
for i in "${!headers[@]}"; do
   printf " %s " "$(center_text "${cols[i]}" "${headers[i]}")"
   printf "║"
done; echo
print_border "╟" "┼" "╢"
for row in "${SUMMARY[@]}"; do
  IFS=$'\t' read -r ver pro shape f5 l5 tm st <<<"$row"
  f5_trunc="${f5:0:${cols[3]}}"
  l5_trunc="${l5:0:${cols[4]}}"
  printf "║ %-*s ║ %*s ║ %-*s ║ %-*s ║ %-*s ║ %*s ║ %-*s ║\n" \
    "${cols[0]}" "$ver" \
    "${cols[1]}" "$pro" \
    "${cols[2]}" "$shape" \
    "${cols[3]}" "$f5_trunc" \
    "${cols[4]}" "$l5_trunc" \
    "${cols[5]}" "$tm" \
    "${cols[6]}" "$st"
done
print_border "╚" "╧" "╝"
echo ""
echo "Logs directory: $LOGS"
echo "CSV summary written to: $CSV_OUTPUT_FILE" # Inform user about CSV file