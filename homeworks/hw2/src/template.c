/*
 * Homework 2 - Sample Template
 * This file is a placeholder source file for Homework 2.
 *
 * Replace this code with your implementation.
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("Homework 2: Hello from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
    
    MPI_Finalize();
    return 0;
}
