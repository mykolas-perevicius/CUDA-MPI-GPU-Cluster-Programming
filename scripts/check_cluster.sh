#!/bin/bash

CONFIG_FILE="config/cluster.conf"
SCRIPT_DIR=$(dirname "$0")

echo "--- Cluster Connectivity Test ---"

# --- Load Configuration ---
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    echo "Please copy config/cluster.conf.template to config/cluster.conf and fill it out."
    exit 1
fi
echo "Loading configuration from $CONFIG_FILE..."
source "$CONFIG_FILE" || { echo "Failed to source config file."; exit 1; }

# Validate essential variables
if [ -z "$MANAGER_HOST" ] || [ -z "$CLUSTER_USER" ] || [ ${#WORKER_HOSTS[@]} -eq 0 ]; then
     echo "Error: MANAGER_HOST, CLUSTER_USER, or WORKER_HOSTS not set in config."
     exit 1
fi

ALL_HOSTS=("$MANAGER_HOST" "${WORKER_HOSTS[@]}")
ALL_IPS=("$MANAGER_IP" "${WORKER_IPS[@]}") # Assuming order matches hosts

# --- Ping Test ---
echo ""
echo "--- Ping Test ---"
PING_FAILED=0
# Assuming IPs are defined and correspond to hosts
if [ ${#ALL_IPS[@]} -ne ${#ALL_HOSTS[@]} ]; then
    echo "Warning: Number of IPs does not match number of hosts in config. Skipping IP ping test."
else
    for ip in "${ALL_IPS[@]}"; do
         if [ -z "$ip" ]; then continue; fi # Skip empty IPs
         echo "Pinging $ip..."
         if ! ping -c 2 "$ip"; then # Send 2 packets
             echo "!!! Ping FAILED for $ip !!!"
             PING_FAILED=1
         fi
    done
fi
[ $PING_FAILED -eq 0 ] && echo "Ping test passed (for defined IPs)."

# --- SSH Test (Passwordless assumed) ---
echo ""
echo "--- SSH Test (Passwordless Login Check) ---"
SSH_FAILED=0
for host in "${ALL_HOSTS[@]}"; do
    echo "Testing SSH to $CLUSTER_USER@$host..."
    # Execute a simple command like 'hostname'
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLUSTER_USER@$host" hostname; then
        echo "!!! SSH FAILED for $CLUSTER_USER@$host !!!"
        echo "    Ensure passwordless SSH is set up correctly."
        SSH_FAILED=1
    fi
done
 [ $SSH_FAILED -eq 0 ] && echo "SSH test passed."


# --- Basic MPI Test ---
echo ""
echo "--- Basic MPI Run Test ---"
MPI_FAILED=0
TEMP_HOSTFILE="hosts.$$.tmp" # Temporary unique hostfile name
echo "Creating temporary hostfile: $TEMP_HOSTFILE"
> "$TEMP_HOSTFILE" # Create or clear the file
for host in "${ALL_HOSTS[@]}"; do
    echo "$host" >> "$TEMP_HOSTFILE"
done

NUM_NODES=${#ALL_HOSTS[@]}
echo "Attempting 'mpirun -hostfile $TEMP_HOSTFILE -np $NUM_NODES hostname'..."
if ! mpirun --oversubscribe -hostfile "$TEMP_HOSTFILE" -np "$NUM_NODES" hostname; then
    echo "!!! Basic MPI run FAILED !!!"
    echo "    Check MPI installation, environment variables (PATH, LD_LIBRARY_PATH), firewall settings, and daemon status (hydra?)."
    MPI_FAILED=1
else
    echo "Basic MPI run seems OK."
fi
rm "$TEMP_HOSTFILE"


# --- Final Summary ---
echo ""
echo "--- Test Summary ---"
if [ $PING_FAILED -eq 0 ] && [ $SSH_FAILED -eq 0 ] && [ $MPI_FAILED -eq 0 ]; then
    echo "Cluster connectivity appears OK."
    exit 0
else
    echo "Cluster connectivity issues DETECTED. Please review logs above."
    exit 1
fi