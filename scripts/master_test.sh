#!/bin/bash
# master_test.sh: Simulates the professor's testing routine.
source ../config/cluster.conf

# Generate a hosts file
HOSTS_FILE=hosts.4
echo "\$MANAGER_IP" > \$HOSTS_FILE
echo "\$WORKER_IP" >> \$HOSTS_FILE

for np in {1..8}; do
    for n in 128 256 512 1024 2048; do
        echo "Testing with np=\$np, n=\$n"
        mpirun -np \$np --hostfile \$HOSTS_FILE --map-by node ../assignments/hw1/src/template \$n
    done
done
