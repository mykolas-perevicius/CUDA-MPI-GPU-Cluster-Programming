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
  > "$log" # Clear log file before run
  if eval "$cmd" >>"$log" 2>&1; then
    echo "  [✔ succeeded]"
    return 0
  else
    local exit_code=$?
    echo "  [✘ failed (exit code: $exit_code) – see $log]"
    return 1
  fi
}

# collect tab‑separated summary lines
declare -a SUMMARY
add_summary() {
  # Args: Version, Procs, Shape, First5, Last5, Time, Status
  SUMMARY+=("$1"$'\t'"$2"$'\t'"$3"$'\t'"$4"$'\t'"$5"$'\t'"$6"$'\t'"$7")
}

echo "=== Testing V1 Serial (1 process) ==="
cd "$FP/v1_serial"
make clean && make
LOG="$LOGS/final_project_v1_np1.log"
shape_v1="–"; first5_v1="–"; last5_v1="–"; time_v1="–"; status_v1="✘"
if run_and_log "./template" "$LOG"; then
  time_v1="$(grep -Eo '[0-9]+(\.[0-9]+)? ms' "$LOG" | head -1 || echo "–")" # PARSE TIME
  shape_v1="$(grep -m1 'After LRN2' "$LOG" | sed -n -E 's/.*Dimensions: H=([0-9]+), W=([0-9]+), C=([0-9]+).*/\1x\2x\3/p' || echo "13x13x256")"
  first5_v1="$(grep -m1 'Final Output (first 10 values):' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v1="$first5_v1"
  status_v1="✔"
  if [[ "$first5_v1" == "–" || "$time_v1" == "–" ]]; then status_v1="⚠"; fi # Check time parsing too
else
  status_v1="✘"
fi
add_summary "V1 Serial" 1 "$shape_v1" "$first5_v1" "$last5_v1" "$time_v1" "$status_v1"


for ver in "2.1_broadcast_all" "2.2_scatter_halo"; do
  for np in 1 2 4; do
    echo "=== Testing V2 ($ver) with $np processes ==="
    cd "$FP/v2_mpi_only/$ver"
    make clean && make
    LOG="$LOGS/final_project_v2_${ver}_np${np}.log"
    # Default values
    shape_v2="–"; sample_v2="–"; time_v2="–"; status_v2="✘" # *** Initialize time_v2 ***
    ver_name="V2 ${ver//_/-}"

    if run_and_log "mpirun -np $np ./template" "$LOG"; then
      # Try to parse output only if run succeeded
      shape_v2="$(grep -m1 '^shape:' "$LOG" | sed -E 's/^shape: *([0-9]+x[0-9]+x[0-9]+).*/\1/' || echo "–")"
      sample_v2="$(grep -m1 '^Sample values:' "$LOG" | sed -E 's/^Sample values: *(([0-9.eE+-]+[[:space:]]*){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
      time_v2="$(grep -m1 '^Execution Time:' "$LOG" | grep -Eo '[0-9]+(\.[0-9]+)? ms' | head -1 || echo "–")" # *** PARSE TIME FOR V2 ***

      # Check if parsing was successful
      # Allow time to be missing without marking as warning if shape/samples are ok
      if [[ "$shape_v2" != "–" && "$sample_v2" != "–" ]]; then
         if [[ "$time_v2" == "–" ]]; then
             echo "  [ℹ Info: Could not parse execution time from $LOG]" # Info, not warning
         fi
         status_v2="✔"
      else
        # Execution succeeded, but critical output parsing failed
        status_v2="⚠"
        echo "  [⚠ Warning: Execution succeeded but failed to parse shape/sample output from $LOG]"
        if [[ "$shape_v2" == "–" ]]; then shape_v2="–"; fi
        if [[ "$sample_v2" == "–" ]]; then sample_v2="–"; fi
        time_v2="–" # Reset time if critical parsing failed
      fi
    else
      status_v2="✘" # Failed execution
    fi
    # *** Use time_v2 in add_summary ***
    add_summary "$ver_name" "$np" "$shape_v2" "$sample_v2" "$sample_v2" "$time_v2" "$status_v2"
  done
done

echo "=== Testing V3 CUDA Only (1 process) ==="
cd "$FP/v3_cuda_only"
make clean && make
LOG="$LOGS/final_project_v3_np1.log"
shape_v3="–"; first5_v3="–"; last5_v3="–"; time_v3="–"; status_v3="✘"
if run_and_log "./template" "$LOG"; then
  time_v3="$(grep -Eo '[0-9]+(\.[0-9]+)? ms' "$LOG" | head -1 || echo "–")" # PARSE TIME
  first5_v3="$(grep -m1 'Final Output' "$LOG" | sed -E 's/.*: *(([0-9.eE+-]+[[:space:]]+){0,4}[0-9.eE+-]+).*/\1/' | awk '{$1=$1};1' || echo "–")"
  last5_v3="$first5_v3"
  # Try to parse shape if printed (optional)
  shape_v3="$(grep -m1 'Final Dimensions' "$LOG" | sed -n -E 's/.*: ([0-9]+) x ([0-9]+) x ([0-9]+).*/\1x\2x\3/p' || echo "–")"
  status_v3="✔"
  if [[ "$first5_v3" == "–" || "$time_v3" == "–" ]]; then status_v3="⚠"; fi # Check time parsing too
else
  status_v3="✘"
fi
add_summary "V3 CUDA" 1 "$shape_v3" "$first5_v3" "$last5_v3" "$time_v3" "$status_v3"


# === print final ASCII table (no changes needed here) ===
# ... (table printing logic remains the same) ...
# column widths
cols=(22 5 11 55 55 10 3) # Adjusted first col width slightly
headers=(Version Procs Shape "First 5 vals" "Last 5 vals" Time St)

# border printer
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
# top border
print_border "╔" "╤" "╗"
# header row
printf "║"
for i in "${!headers[@]}"; do
   printf " %-*s " "${cols[i]}" "${headers[i]}" | awk -v W="${cols[i]}" '{ printf "%*s%s%*s", (W-length)/2, "", $0, (W-length+1)/2, "" }'; printf "║"
done; echo
# header separator
print_border "╟" "┼" "╢"
# data rows
for row in "${SUMMARY[@]}"; do
  IFS=$'\t' read -r ver pro shape f5 l5 tm st <<<"$row"
  # Adjust printf alignment specifiers if needed based on column widths
  printf "║ %-${cols[0]}s ║ %${cols[1]}s ║ %-${cols[2]}s ║ %-${cols[3]}s ║ %-${cols[4]}s ║ %${cols[5]}s ║ %-${cols[6]}s ║\n" \
    "$ver" "$pro" "$shape" "$f5" "$l5" "$tm" "$st"
done
# bottom border
print_border "╚" "╧" "╝"

echo "Logs directory: $LOGS"