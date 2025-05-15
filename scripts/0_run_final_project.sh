#!/usr/bin/env bash
set -euo pipefail

# --- Script Setup ---
ROOT_DIR_0="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/common_test_utils.sh
source "$ROOT_DIR_0/scripts/common_test_utils.sh" # Source common functions

# --- Initialize Global Variables defined in common_test_utils.sh ---
ROOT="$ROOT_DIR_0"
FP="$ROOT/final_project"
LOGS_BASE_DIR="$FP/logs" # Central logs directory
mkdir -p "$LOGS_BASE_DIR"

MACHINE_ID_RAW=$(hostname -s)
MACHINE_ID_CLEAN="${MACHINE_ID_RAW//[^a-zA-Z0-9_-]/_}"
SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Script specific prefix for session ID
SESSION_ID="script0_session_${SESSION_TIMESTAMP}_${MACHINE_ID_CLEAN}"
SESSION_LOG_DIR="$LOGS_BASE_DIR/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"

CSV_SUMMARY_FILE="$SESSION_LOG_DIR/summary_report_${SESSION_ID}.csv"
GIT_COMMIT_SHORT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")

# --- Header ---
echo "================================================================================"
echo "Legacy Test Script 0: Basic Project Runner"
echo "Session: $SESSION_ID"
echo "Machine: $MACHINE_ID_CLEAN, Git Commit: $GIT_COMMIT_SHORT"
echo "Logs: $SESSION_LOG_DIR"
echo "CSV Summary: $CSV_SUMMARY_FILE"
echo "================================================================================"

# --- Call CUDA Arch Detection ---
# This function is defined in common_test_utils.sh and exports DETECTED_GPU_ARCH_FLAGS
_detect_and_set_cuda_arch_flags

# --- Write CSV Header ---
# (Ensure this matches the one in common_test_utils.sh if you customize it there)
echo "SessionID,MachineID,GitCommit,EntryTimestamp,ProjectVariant,NumProcesses,MakeLogFile,BuildSucceeded,BuildMessage,RunLogFile,RunCommandSucceeded,RunEnvironmentWarning,RunMessage,ParseSucceeded,ParseMessage,OverallStatusSymbol,OverallStatusMessage,ExecutionTime_ms,OutputShape,OutputFirst5Values" > "$CSV_SUMMARY_FILE"


# --- Test V1 Serial ---
if setup_test_case "V1 Serial" 1 "v1_serial" "v1_serial"; then
    execute_single_test_case
fi

# --- Test V2 MPI Only ---
for ver_subdir in "2.1_broadcast_all" "2.2_scatter_halo"; do
  ver_tag_suffix="${ver_subdir//./_}" # e.g., 2_1_broadcast_all
  ver_display_name_suffix="${ver_subdir//_/-}" # e.g., 2.1-broadcast-all
  for np_val in 1 2 4; do
    if setup_test_case "V2 $ver_display_name_suffix" "$np_val" "v2_mpi_only/$ver_subdir" "v2_${ver_tag_suffix}"; then
        execute_single_test_case
    fi
  done
done

# --- Test V3 CUDA Only ---
if setup_test_case "V3 CUDA Only" 1 "v3_cuda_only" "v3_cuda"; then
    execute_single_test_case
fi

# --- Test V4 MPI+CUDA ---
for np_val in 1 2 4; do
    if setup_test_case "V4 MPI+CUDA" "$np_val" "v4_mpi_cuda" "v4_mpi_cuda"; then
        execute_single_test_case
    fi
done

# --- Print Final ASCII Table ---
_print_summary_table

echo "================================================================================"
echo "Legacy Test Script 0 Finished. Session: $SESSION_ID"
echo "================================================================================"