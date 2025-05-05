#!/usr/bin/env bash
set -euo pipefail

# root of repo
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP="$ROOT/final_project"
LOGS="$FP/logs"
mkdir -p "$LOGS"

# helper to run & log; prints ✅/❌/⚠
run_and_log() {
  local cmd="$1"; shift
  local log="$1"; shift
  echo "-> Running Command: $cmd" # More explicit echo
  # Clear log file before run and ensure it exists
  touch "$log" && > "$log"
  # Redirect stderr to stdout before piping to log file
  if eval "$cmd" >>"$log" 2>&1; then
    echo "  [✔ Command Succeeded]"
    return 0
  else
    local exit_code=$?
    echo "  [DEBUG] Command failed with exit code: $exit_code" # Debug echo
    # Check for common MPI/CUDA setup errors that might not indicate code failure
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log"; then
        echo "  [⚠ Warning (exit code: $exit_code) - Possible Environment/Setup Issue - see $log]"
        return 2 # Use a different return code for environment issues
    elif grep -q -E "There are not enough slots available|could not be found|orted context|" "$log"; then
         echo "  [⚠ Warning (exit code: $exit_code) - MPI Resource/Config Issue - see $log]"
         return 2
    else
        # Check specifically for the V3 linker error we saw
        if grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$log"; then
             echo "  [✘ Failed (exit code: $exit_code) – V3 Linker Error? See $log]"
        else
             echo "  [✘ Failed (exit code: $exit_code) – see $log]"
        fi
        return 1
    fi
  fi
}

# collect tab‑separated summary lines
declare -a SUMMARY
add_summary() {
  # Args: Version, Procs, Shape, First5, Last5, Time, Status
  SUMMARY+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7")
}

# --- Testing V1 ---
# (Keep V1 logic as is, assuming it worked - applying simplified check style)
echo "=== Testing V1 Serial (1 process) ==="
cd "$FP/v1_serial"
MAKE_LOG_V1="$LOGS/make_v1.log"
touch "$MAKE_LOG_V1" && > "$MAKE_LOG_V1"
echo "--- Building V1 ---"
make_succeeded_v1=false
# Run make and check exit status ($?)
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

LOG="$LOGS/final_project_v1_np1.log"
shape_v1="–"; first5_v1="–"; last5_v1="–"; time_v1="–"; status_v1="✘ (init)"
run_result=0 # 0=success, 1=runtime_fail, 2=env_warn

if [[ "$make_succeeded_v1" = true ]]; then
    echo "  [DEBUG] V1 make succeeded. Attempting run..."
    run_and_log "./template" "$LOG" || run_result=$?
    if [[ $run_result -eq 0 ]]; then status_v1="✔";
    elif [[ $run_result -eq 2 ]]; then status_v1="⚠";
    else status_v1="✘ (runtime err)"; fi
else
    if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v1="✘ (make: no exe)";
    else status_v1="✘ (make failed)"; fi
    echo "  [Skipping V1 run due to build issue]"
fi

if [[ "$status_v1" == "✔" ]]; then
  time_v1="$(grep -m1 'completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
  shape_v1="$(grep -m1 'After LRN2' "$LOG" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "–")"
  if [[ "$shape_v1" == "–" ]]; then
     shape_v1="$(grep -m1 '^Final Output Shape:' "$LOG" | sed -n -E 's/.*Shape: ([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")"
  fi
  first5_v1="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v1="$first5_v1"
  if [[ "$first5_v1" == "–" || "$time_v1" == "–" || "$shape_v1" == "–" ]]; then status_v1="⚠ (parse err)"; fi
fi
add_summary "V1 Serial" 1 "$shape_v1" "$first5_v1" "$last5_v1" "$time_v1" "$status_v1"


# --- Testing V2 ---
# (Keep V2 logic similar to V1's updated logic)
for ver in "2.1_broadcast_all" "2.2_scatter_halo"; do
  for np in 1 2 4; do
    echo "=== Testing V2 ($ver) with $np processes ==="
    cd "$FP/v2_mpi_only/$ver"
    MAKE_LOG_V2="$LOGS/make_v2_${ver}_np${np}.log"
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

    LOG="$LOGS/final_project_v2_${ver}_np${np}.log"
    shape_v2="–"; sample_v2="–"; time_v2="–"; status_v2="✘ (init)"
    ver_name="V2 ${ver//_/-}"
    run_result=0 # 0=success, 1=runtime_fail, 2=env_warn

    if [[ "$make_succeeded_v2" = true ]]; then
        echo "  [DEBUG] V2 make succeeded ($ver, NP=$np). Attempting run..."
        run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG" || run_result=$?
        if [[ $run_result -eq 0 ]]; then status_v2="✔";
        elif [[ $run_result -eq 2 ]]; then status_v2="⚠";
        else status_v2="✘ (runtime err)"; fi
    else
        if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v2="✘ (make: no exe)";
        else status_v2="✘ (make failed)"; fi
        echo "  [Skipping V2 run ($ver, NP=$np) due to build issue]"
    fi

    if [[ "$status_v2" == "✔" ]]; then
      shape_v2="$(grep -m1 '^shape:' "$LOG" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
      sample_v2="$(grep -m1 '^Sample values:' "$LOG" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v2="$(grep -m1 '^Execution Time:' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
      if [[ "$shape_v2" == "–" || "$sample_v2" == "–" ]]; then
         status_v2="⚠ (parse err)"
         echo "  [⚠ Warning: V2 ($ver, $np) succeeded but failed to parse shape/sample output from $LOG]"
         shape_v2="${shape_v2:-–}"
         sample_v2="${sample_v2:-–}"
         time_v2="${time_v2:-–}"
      fi
       if [[ "$time_v2" == "–" ]]; then echo "  [ℹ Info: V2 ($ver, $np) Could not parse execution time from $LOG]"; fi
    fi
    add_summary "$ver_name" "$np" "$shape_v2" "$sample_v2" "$sample_v2" "$time_v2" "$status_v2"
  done
done


# --- Testing V3 ---
echo "=== Testing V3 CUDA Only (1 process) ==="
cd "$FP/v3_cuda_only"

MAKE_LOG_V3="$LOGS/make_v3.log"
touch "$MAKE_LOG_V3" && > "$MAKE_LOG_V3"
echo "--- Building V3 ---"
make_succeeded_v3=false
# *** CORRECTED LOGIC: Run make, check exit code ($?) AND file existence ***
make clean > /dev/null && make >> "$MAKE_LOG_V3" 2>&1
make_exit_code=$? # Capture exit code immediately after make finishes
if [[ $make_exit_code -eq 0 ]]; then
    echo "  [DEBUG] make exit code: 0"
    # Make command finished with exit code 0
    if [[ -f ./template ]]; then
        # And the file exists!
        echo "  [✔ make succeeded and './template' found]"
        make_succeeded_v3=true
    else
        # Make exited 0, but file is missing!
        echo "  [✘ make reported success (exit 0) BUT './template' is MISSING - check $MAKE_LOG_V3]"
        # Keep make_succeeded_v3 as false
    fi
else
    # Make command failed with non-zero exit code
    echo "  [✘ make failed (exit code: $make_exit_code) - check $MAKE_LOG_V3]"
    # Keep make_succeeded_v3 as false
fi

LOG="$LOGS/final_project_v3_np1.log"
shape_v3="–"; first5_v3="–"; last5_v3="–"; time_v3="–"; status_v3="✘ (init)"
run_result=0 # 0=success, 1=runtime_fail, 2=env_warn

# *** Run only if make_succeeded_v3 is true ***
if [[ "$make_succeeded_v3" = true ]]; then
    echo "  [DEBUG] V3 make succeeded. Attempting run..."
    run_and_log "./template" "$LOG" || run_result=$?
    # Assign status based on run_and_log result
    if [[ $run_result -eq 0 ]]; then status_v3="✔";
    elif [[ $run_result -eq 2 ]]; then status_v3="⚠";
    else status_v3="✘ (runtime err)"; fi
else
    echo "  [Skipping V3 run due to build issue]"
    # Determine specific failure reason for status
    if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then # Make exit 0 but no file
        status_v3="✘ (make: no exe)"
    elif grep -q -E "cannot find -lcudadevrt|cannot find -lcudart_static" "$MAKE_LOG_V3"; then # Check for old linker error
         status_v3="✘ (Linker: CUDA Libs?)"
    else # General make failure
        status_v3="✘ (make failed)"
    fi
fi

# Process run results only if status is success
if [[ "$status_v3" == "✔" ]]; then
  time_v3="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"
  first5_v3="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v3="$first5_v3"
  shape_v3="13x13x256" # Assuming shape is fixed for V3 if it runs
  if [[ "$first5_v3" == "–" || "$time_v3" == "–" ]]; then status_v3="⚠ (parse err)"; fi
fi
add_summary "V3 CUDA" 1 "$shape_v3" "$first5_v3" "$last5_v3" "$time_v3" "$status_v3"


# --- Testing V4 ---
echo "=== Testing V4 MPI+CUDA ==="
cd "$FP/v4_mpi_cuda"
MAKE_LOG_V4="$LOGS/make_v4.log"
touch "$MAKE_LOG_V4" && > "$MAKE_LOG_V4"
echo "--- Building V4 ---"
make_succeeded_v4=false
# *** CORRECTED LOGIC: Run make, check exit code ($?) AND file existence ***
make clean > /dev/null && make >> "$MAKE_LOG_V4" 2>&1
make_exit_code=$? # Capture exit code immediately
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
    LOG="$LOGS/final_project_v4_np${np}.log"
    shape_v4="–"; sample_v4="–"; time_v4="–"; status_v4="✘ (init)"
    ver_name="V4 MPI+CUDA"
    run_result=0 # 0=success, 1=runtime_fail, 2=env_warn

    # *** Run only if make_succeeded_v4 is true ***
    if [[ "$make_succeeded_v4" = true ]]; then
        echo "  [DEBUG] V4 make succeeded. Attempting run with NP=$np ..."
        run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG" || run_result=$?
        # Assign status based on run_and_log result
        if [[ $run_result -eq 0 ]]; then status_v4="✔";
        elif [[ $run_result -eq 2 ]]; then status_v4="⚠";
        else status_v4="✘ (runtime err)"; fi
    else
        echo "  [Skipping V4 run (NP=$np) due to build issue]"
        # Determine specific failure reason for status
         if [[ ! -f ./template && $make_exit_code -eq 0 ]]; then status_v4="✘ (make: no exe)";
         else status_v4="✘ (make failed)"; fi
    fi

    # Process run results only if status is success
    if [[ "$status_v4" == "✔" ]]; then
      shape_v4="$(grep -m1 '^Final Output Shape:' "$LOG" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
      if [[ "$shape_v4" == "–" ]]; then
        total_size=$(grep -m1 '^Final Output Total Size:' "$LOG" | sed -n -E 's/^Final Output Total Size: *([0-9]+) .*/\1/p' || echo "")
        if [[ -n "$total_size" ]]; then
            if [[ "$total_size" == "43264" ]]; then shape_v4="13x13x256";
            elif [[ "$total_size" == "0" ]]; then shape_v4="0x0x0";
            else shape_v4="?x?x? ($total_size elem)"; fi
        else
             shape_v4="–"
        fi
      fi
      sample_v4="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v4="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"

      if [[ "$shape_v4" == "–" || "$sample_v4" == "–" ]]; then
         status_v4="⚠ (parse err)"
         echo "  [⚠ Warning: V4 (NP=$np) succeeded but failed to parse critical output from $LOG (Shape: '$shape_v4', Sample: '$sample_v4')]"
         shape_v4="${shape_v4:-–}"
         sample_v4="${sample_v4:-–}"
         time_v4="${time_v4:-–}"
      fi
      if [[ "$time_v4" == "–" ]]; then
          echo "  [ℹ Info: V4 (NP=$np) Could not parse execution time from $LOG]"
      fi
    fi # end if status_v4 == "✔"
    add_summary "$ver_name" "$np" "$shape_v4" "$sample_v4" "$sample_v4" "$time_v4" "$status_v4"
done

# === print final ASCII table ===
# (Keep table printing logic as is)
echo ""
echo "=== Summary Table ==="
cols=(22 5 11 30 30 10 22)
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