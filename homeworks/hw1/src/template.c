/*
 * Mat Mult
 * CS485 GPU cluster programming - MPI + CUDA
 * Spring 2025
 * template for HW1 - 3
 * HW1 - point to point communication
 * HW2 - collective communication
 * HW3 - one-sided communication
 * Andrew Sohn
 *
 * HW1 Solution using Point-to-Point (Send/Recv) and Broadcast
 * (CppCheck suggestions applied)
 */

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h> // For srand
 
 #define MAXDIM (1 << 12) /* 4096 */
 #define ROOT 0
 
 // --- Function Prototypes ---
 // Added 'const' where applicable based on CppCheck suggestions
 int mat_mult(const double *a, const double *b, double *c, int n, int my_work);
 void init_data(double *data, int data_size);
 int check_result(const double *C, const double *D, int n);
 
 // --- Main Function ---
 int main(int argc, char *argv[]) {
   int n = 64, n_sq, my_work;
   int my_rank, num_procs = 1;
   double *A = NULL, *B = NULL, *C = NULL, *D = NULL; /* D is for local computation result */
   double *local_A = NULL, *local_C = NULL; /* Buffers for local work */
   int elms_to_comm;
   // MPI_Status status; // Removed as MPI_STATUS_IGNORE is used
 
   MPI_Comm world = MPI_COMM_WORLD;
 
   MPI_Init(&argc, &argv);
 
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
 
   // --- Input Processing ---
   if (argc > 1) {
     n = atoi(argv[1]);
     if (n <= 0 || (n & (n - 1)) != 0) { // Check if n is positive power of 2
       if (my_rank == ROOT)
         fprintf(stderr,
                 "Error: Matrix dimension 'n' (%d) must be a positive power "
                 "of two.\n",
                 n);
       MPI_Abort(world, 1);
     }
     if (n > MAXDIM) {
       if (my_rank == ROOT)
         fprintf(stderr,
                 "Warning: Matrix dimension 'n' (%d) exceeds MAXDIM (%d). "
                 "Clamping to MAXDIM.\n",
                 n, MAXDIM);
       n = MAXDIM;
     }
   }
    if (n % num_procs != 0) {
      if (my_rank == ROOT)
        fprintf(stderr,
                "Error: Matrix dimension 'n' (%d) must be divisible by the "
                "number of processes (%d).\n",
                n, num_procs);
      MPI_Abort(world, 1);
    }
 
   n_sq = n * n;
   my_work = n / num_procs;      // Rows per process
   elms_to_comm = my_work * n; // Elements per process for A and C chunks
 
   // --- Memory Allocation (Removed C-style casts) ---
   B = malloc(sizeof(double) * n_sq);
   if (B == NULL) {
     perror("Failed to allocate B");
     MPI_Abort(world, 1);
   }
 
   if (my_rank == ROOT) {
     A = malloc(sizeof(double) * n_sq);
     C = malloc(sizeof(double) * n_sq);
     D = malloc(sizeof(double) * n_sq);
     if (A == NULL || C == NULL || D == NULL) {
       perror("Failed to allocate A, C or D on ROOT");
       MPI_Abort(world, 1);
     }
     local_A = A; // Root uses the start of its A
     local_C = C; // Root uses the start of its C
     printf("pid=%d: num_procs=%d n=%d my_work=%d (rows per proc)\n", my_rank,
            num_procs, n, my_work);
   } else {
     local_A = malloc(sizeof(double) * elms_to_comm); // my_work rows of A
     local_C = malloc(sizeof(double) * elms_to_comm); // my_work rows of C
     if (local_A == NULL || local_C == NULL) {
       perror("Failed to allocate local_A or local_C");
       MPI_Abort(world, 1);
     }
   }
 
   // --- Data Initialization (Root Only) ---
   if (my_rank == ROOT) {
     srand(time(NULL)); // Seed random number generator once
     init_data(A, n_sq);
     init_data(B, n_sq); // Initialize B on Root before broadcast
   }
 
   // --- Start Timer ---
   MPI_Barrier(world); // Synchronize before starting timer
   // Scope reduction: Moved timer variables closer to use
   double start_time = MPI_Wtime();
 
   // --- Distribute Data ---
 
   // 1. Scatter A using Point-to-Point Send/Recv
   if (my_rank == ROOT) {
     for (int p = 1; p < num_procs; p++) {
       int start_index = p * elms_to_comm; // Start element index in full A
       MPI_Send(&A[start_index], elms_to_comm, MPI_DOUBLE, p, 0, world);
     }
   } else {
     MPI_Recv(local_A, elms_to_comm, MPI_DOUBLE, ROOT, 0, world,
              MPI_STATUS_IGNORE);
   }
 
   // 2. Broadcast B to all processes
   MPI_Bcast(B, n_sq, MPI_DOUBLE, ROOT, world);
 
   // --- Local Computation ---
   mat_mult(local_A, B, local_C, n, my_work);
 
   // --- Gather Results ---
   if (my_rank == ROOT) {
     for (int p = 1; p < num_procs; p++) {
       int start_index = p * elms_to_comm; // Start element index in full C
       MPI_Recv(&C[start_index], elms_to_comm, MPI_DOUBLE, p, 0, world,
                MPI_STATUS_IGNORE);
     }
   } else {
     MPI_Send(local_C, elms_to_comm, MPI_DOUBLE, ROOT, 0, world);
   }
 
   // --- Stop Timer and Verify (Root Only) ---
   if (my_rank == ROOT) {
     // Scope reduction: Moved timer variables closer to use
     double end_time = MPI_Wtime();
     double elapsed = end_time - start_time;
 
     printf("pid=%d: Parallel computation finished in %f seconds.\n", my_rank,
            elapsed);
 
     printf("pid=%d: Performing serial computation for verification...\n", my_rank);
     double serial_start = MPI_Wtime();
     mat_mult(A, B, D, n, n); // Use full n for serial version
     double serial_end = MPI_Wtime();
     printf("pid=%d: Serial computation finished in %f seconds.\n", my_rank,
            serial_end - serial_start);
 
     // Scope reduction: Moved flag closer to use
     int flag = check_result(C, D, n);
     if (flag) {
       printf("--------------------------------------\n");
       printf("pid=%d: Test: FAILED\n", my_rank);
       printf("--------------------------------------\n");
     } else {
       printf("--------------------------------------\n");
       printf("pid=%d: Test: PASSED\n", my_rank);
       printf("pid=%d: Total PARALLEL time: %f seconds.\n", my_rank, elapsed);
       printf("--------------------------------------\n");
     }
   }
 
   // --- Cleanup ---
   if (my_rank == ROOT) {
     free(A);
     free(C);
     free(D);
   } else {
     free(local_A);
     free(local_C);
   }
   free(B); // All processes free B
 
   MPI_Finalize();
   return 0;
 }
 
 // --- Matrix Multiplication Function ---
 // Added const qualifiers
 int mat_mult(const double *a, const double *b, double *c, int n, int my_work) {
   int i, j, k;
   double sum;
   for (i = 0; i < my_work; i++) {
     for (j = 0; j < n; j++) {
       sum = 0.0;
       for (k = 0; k < n; k++) {
         sum += a[i * n + k] * b[k * n + j];
       }
       c[i * n + j] = sum;
     }
   }
   return 0;
 }
 
 // --- Initialize an array with random double data ---
 void init_data(double *data, int data_size) {
   int i;
   for (i = 0; i < data_size; i++) {
     data[i] = (double)(rand() % 10); // Simple integers 0-9 as doubles
   }
 }
 
 // --- Compare two double matrices C and D ---
 // Added const qualifiers
 int check_result(const double *C, const double *D, int n) {
   int i, j, flag = 0;
   double diff, tolerance = 1e-6;
 
   for (i = 0; i < n; i++) {
     for (j = 0; j < n; j++) {
       diff = C[i * n + j] - D[i * n + j];
       if (diff > tolerance || diff < -tolerance) {
         if (flag == 0) { // Print only first mismatch
           printf("ERROR: Mismatch found at C[%d][%d]=%f != D[%d][%d]=%f (Diff: %f)\n",
                  i, j, C[i * n + j], i, j, D[i * n + j], diff);
         }
         flag = 1;
         // return flag; // Optional: Exit early on first mismatch
       }
     }
   }
   return flag; // Return 0 if no mismatch, 1 otherwise
 }
 
 /*
   End of file
  */