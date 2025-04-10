#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h> // For timing, if needed

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Optional: Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    if (rank == 0) {
        printf("MPI World Size: %d\n", size);
    }

    printf("Hello from processor %s, rank %d out of %d processors\n",
           processor_name, rank, size);

    // --- Your Homework Logic Starts Here ---

    // Example: Handle command line argument for problem size
    int problem_size = 0;
    if (argc > 1) {
        problem_size = atoi(argv[1]);
        if (rank == 0) {
             printf("Problem size input: %d\n", problem_size);
        }
    } else {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <num_procs> %s <problem_size>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Exit if no size provided
    }


    // --- Your Homework Logic Ends Here ---


    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}