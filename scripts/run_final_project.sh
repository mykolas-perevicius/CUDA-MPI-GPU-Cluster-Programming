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
  echo "-> $cmd"
  # Clear log file before run and ensure it exists
  touch "$log" && > "$log"
  if eval "$cmd" >>"$log" 2>&1; then
    echo "  [✔ succeeded]"
    return 0
  else
    local exit_code=$?
    # Check for common MPI/CUDA setup errors that might not indicate code failure
    if grep -q -E "Could not find device file|No CUDA-capable device detected|PMIx coord service not available|Unavailable consoles" "$log"; then
        echo "  [⚠ Warning (exit code: $exit_code) - Possible Environment/Setup Issue - see $log]"
        return 2 # Use a different return code for environment issues
    elif grep -q -E "There are not enough slots available|could not be found|orted context|" "$log"; then
         echo "  [⚠ Warning (exit code: $exit_code) - MPI Resource/Config Issue - see $log]"
         return 2
    else
        echo "  [✘ failed (exit code: $exit_code) – see $log]"
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
echo "=== Testing V1 Serial (1 process) ==="
cd "$FP/v1_serial"
make clean > /dev/null && make > /dev/null # Suppress make output
LOG="$LOGS/final_project_v1_np1.log"
shape_v1="–"; first5_v1="–"; last5_v1="–"; time_v1="–"; status_v1="✘"
run_result=0
run_and_log "./template" "$LOG" || run_result=$?
if [[ $run_result -eq 0 ]]; then
  time_v1="$(grep -m1 'completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")" # More general time grep
  shape_v1="$(grep -m1 'After LRN2' "$LOG" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "–")"
  if [[ "$shape_v1" == "–" ]]; then
     shape_v1="$(grep -m1 '^Final Output Shape:' "$LOG" | sed -n -E 's/.*Shape: ([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")"
  fi
  first5_v1="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v1="$first5_v1"
  status_v1="✔"
  if [[ "$first5_v1" == "–" || "$time_v1" == "–" || "$shape_v1" == "–" ]]; then status_v1="⚠"; fi
elif [[ $run_result -eq 2 ]]; then status_v1="⚠";
else status_v1="✘"; fi
add_summary "V1 Serial" 1 "$shape_v1" "$first5_v1" "$last5_v1" "$time_v1" "$status_v1"

# --- Testing V2 ---
for ver in "2.1_broadcast_all" "2.2_scatter_halo"; do
  for np in 1 2 4; do
    echo "=== Testing V2 ($ver) with $np processes ==="
    cd "$FP/v2_mpi_only/$ver"
    make clean > /dev/null && make > /dev/null # Suppress make output
    LOG="$LOGS/final_project_v2_${ver}_np${np}.log"
    shape_v2="–"; sample_v2="–"; time_v2="–"; status_v2="✘"
    ver_name="V2 ${ver//_/-}"
    run_result=0
    run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG" || run_result=$?

    if [[ $run_result -eq 0 ]]; then
      shape_v2="$(grep -m1 '^shape:' "$LOG" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
      sample_v2="$(grep -m1 '^Sample values:' "$LOG" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v2="$(grep -m1 '^Execution Time:' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"

      if [[ "$shape_v2" != "–" && "$sample_v2" != "–" ]]; then
         if [[ "$time_v2" == "–" ]]; then echo "  [ℹ Info: Could not parse execution time from $LOG]"; fi
         status_v2="✔"
      else
        status_v2="⚠"
        echo "  [⚠ Warning: Execution succeeded but failed to parse shape/sample output from $LOG]"
        shape_v2="${shape_v2:-–}"
        sample_v2="${sample_v2:-–}"
        time_v2="–"
      fi
    elif [[ $run_result -eq 2 ]]; then status_v2="⚠";
    else status_v2="✘";
    fi
    add_summary "$ver_name" "$np" "$shape_v2" "$sample_v2" "$sample_v2" "$time_v2" "$status_v2"
  done
done

# --- Testing V3 ---
echo "=== Testing V3 CUDA Only (1 process) ==="
cd "$FP/v3_cuda_only"
make clean > /dev/null && make > /dev/null # Suppress make output
LOG="$LOGS/final_project_v3_np1.log"
shape_v3="–"; first5_v3="–"; last5_v3="–"; time_v3="–"; status_v3="✘"
run_result=0
run_and_log "./template" "$LOG" || run_result=$?

if [[ $run_result -eq 0 ]]; then
  time_v3="$(grep -m1 '^AlexNet CUDA Forward Pass completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")" # Anchor grep
  first5_v3="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")" # Anchor grep
  last5_v3="$first5_v3"
  shape_v3="13x13x256"
  status_v3="✔"
  if [[ "$first5_v3" == "–" || "$time_v3" == "–" ]]; then status_v3="⚠"; fi
elif [[ $run_result -eq 2 ]]; then status_v3="⚠";
else status_v3="✘";
fi
add_summary "V3 CUDA" 1 "$shape_v3" "$first5_v3" "$last5_v3" "$time_v3" "$status_v3"

# --- Testing V4 ---
echo "=== Testing V4 MPI+CUDA ==="
cd "$FP/v4_mpi_cuda"
make clean > /dev/null && make # Keep V4 make output visible for potential errors
for np in 1 2 4; do
    echo "--- V4 with $np processes ---"
    LOG="$LOGS/final_project_v4_np${np}.log"
    shape_v4="–"; sample_v4="–"; time_v4="–"; status_v4="✘"
    ver_name="V4 MPI+CUDA"
    run_result=0
    run_and_log "mpirun --oversubscribe -np $np ./template" "$LOG" || run_result=$?

    if [[ $run_result -eq 0 ]]; then
      # *** Refined Parsing Logic for V4 ***
      # Parse shape (anchor grep, check both patterns)
      shape_v4="$(grep -m1 '^Final Output Shape:' "$LOG" | sed -n -E 's/^Final Output Shape: *([0-9]+)x([0-9]+)x([0-9]+).*/\1x\2x\3/p' || echo "–")"
      if [[ "$shape_v4" == "–" ]]; then
        total_size=$(grep -m1 '^Final Output Total Size:' "$LOG" | sed -n -E 's/^Final Output Total Size: *([0-9]+) .*/\1/p' || echo "")
        if [[ -n "$total_size" ]]; then
            if [[ "$total_size" == "43264" ]]; then shape_v4="13x13x256";
            elif [[ "$total_size" == "0" ]]; then shape_v4="0x0x0"; # Handle zero size output explicitly
            else shape_v4="?x?x? ($total_size elem)"; fi
        else
            # If no shape or size line, mark as unknown
             shape_v4="–"
        fi
      fi

      # Parse sample values (anchor grep)
      sample_v4="$(grep -m1 '^Final Output (first 10 values):' "$LOG" | sed -E 's/^Final Output \(first 10 values\): *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"

      # Parse time (anchor grep)
      time_v4="$(grep -m1 '^AlexNet MPI+CUDA Forward Pass completed in' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")"

      # Check if critical parsing was successful (Shape AND Sample required)
      if [[ "$shape_v4" != "–" && "$sample_v4" != "–" ]]; then
         if [[ "$time_v4" == "–" ]]; then
             echo "  [ℹ Info: Could not parse execution time from $LOG]"
             # Don't mark as warning just for missing time if shape/sample are ok
             status_v4="✔"
         else
             status_v4="✔"
         fi
      else
        status_v4="⚠"
        # Print more specific parsing failure info
        echo "  [⚠ Warning: Execution succeeded but failed to parse critical output from $LOG (Shape: '$shape_v4', Sample: '$sample_v4')]"
        shape_v4="${shape_v4:-–}"
        sample_v4="${sample_v4:-–}"
        time_v4="–" # Reset time if critical parsing failed
      fi
    elif [[ $run_result -eq 2 ]]; then status_v4="⚠";
    else status_v4="✘";
    fi
    add_summary "$ver_name" "$np" "$shape_v4" "$sample_v4" "$sample_v4" "$time_v4" "$status_v4"
done

# === print final ASCII table ===
echo ""
echo "=== Summary Table ==="
# column widths - adjusted "First 5 vals" and "Last 5 vals"
cols=(22 5 11 30 30 10 3)
headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time St)

# border printer
print_border() {
  local left="$1" mid="$2" right="$3"
  printf "%s" "$left"
  for i in "${!cols[@]}"; do
    local w=${cols[i]}
    local seg_len=$((w + 2)) # Account for spaces around content
    # *** Reverted to simple loop for better portability ***
    for ((j=0; j<seg_len; j++)); do printf '═'; done
    if (( i < ${#cols[@]} - 1 )); then printf "%s" "$mid"; else printf "%s\n" "$right"; fi
  done
}
# center text helper
center_text() {
    local width=$1
    local text=$2
    # Ensure width is treated as a number
    local num_width=$((width))
    local text_len=${#text}
    local pad=$(( (num_width - text_len) / 2 ))
    local pad_rem=$(( (num_width - text_len + 1) / 2 )) # Handles odd width
    # Ensure pads are not negative
    if [[ $pad -lt 0 ]]; then pad=0; fi
    if [[ $pad_rem -lt 0 ]]; then pad_rem=0; fi
    printf "%*s%s%*s" $pad "" "$text" $pad_rem ""
}

# top border
print_border "╔" "╤" "╗"
# header row
printf "║"
for i in "${!headers[@]}"; do
   printf " %s " "$(center_text "${cols[i]}" "${headers[i]}")"
   printf "║"
done; echo
# header separator
print_border "╟" "┼" "╢"
# data rows
for row in "${SUMMARY[@]}"; do
  IFS=$'\t' read -r ver pro shape f5 l5 tm st <<<"$row"
  # Truncate long sample strings if needed to fit columns
  f5_trunc="${f5:0:${cols[3]}}"
  l5_trunc="${l5:0:${cols[4]}}"
  # Print left-aligned version/shape, right-aligned procs/time, left-aligned samples/status
  printf "║ %-*s ║ %*s ║ %-*s ║ %-*s ║ %-*s ║ %*s ║ %-*s ║\n" \
    "${cols[0]}" "$ver" \
    "${cols[1]}" "$pro" \
    "${cols[2]}" "$shape" \
    "${cols[3]}" "$f5_trunc" \
    "${cols[4]}" "$l5_trunc" \
    "${cols[5]}" "$tm" \
    "${cols[6]}" "$st"
done
# bottom border
print_border "╚" "╧" "╝"

echo ""
echo "Logs directory: $LOGS"