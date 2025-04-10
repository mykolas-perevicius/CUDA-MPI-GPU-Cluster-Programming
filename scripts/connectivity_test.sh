#!/bin/bash
# connectivity_test.sh: Tests SSH, ping, and a simple MPI run.
source ../config/cluster.conf

echo "Testing SSH connectivity to manager node (\$MANAGER_IP)..."
ssh \$MANAGER_IP "hostname" || { echo "SSH to manager failed"; exit 1; }

echo "Testing SSH connectivity to worker node (\$WORKER_IP)..."
ssh \$WORKER_IP "hostname" || { echo "SSH to worker failed"; exit 1; }

echo "Pinging manager node (\$MANAGER_IP)..."
ping -c 2 \$MANAGER_IP || { echo "Ping to manager failed"; exit 1; }

echo "Pinging worker node (\$WORKER_IP)..."
ping -c 2 \$WORKER_IP || { echo "Ping to worker failed"; exit 1; }

echo "Running MPI hello world test..."
mpirun -np 2 --host \$MANAGER_IP,\$WORKER_IP ../assignments/hw1/src/template || { echo "MPI test failed"; exit 1; }

echo "Connectivity tests passed."
