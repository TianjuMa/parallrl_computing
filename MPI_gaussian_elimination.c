/*Gaussian elimination without pivoting */

/* Algorithm description:                                                                   
  
  Use three pointers to specified each arrays, inside the main function, doing the 
  initialize and array allocate, then doing the gaussian elimilation inside gauss function.
  
  In gauss function, read the start time before doing the elimilation. After the first loop,
  broadcast the A[norm] and B[norm] to each processes, and inside the process 0 send the A[row] 
  and B[row] to each process by static interleaving. Then process 0 doing elimilation, after that
  receive the result from other processes. At other processes, receive the A[row] and B[row] in
  each row elimilation then doing the task, sending the results to process 0, by static interleaving 
  format. At each norm, MPI_Barrier here to make sure that the synchronization.

  After gauss, we let the process 0 to do the backSubstitude, which do not need to do the parallel. 

*/
/* Tianju Ma, Yuhang Peng */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <mpi.h>

#define MAXN 8000 /* Max value of N */
int N; /* Matrix size */
#define DIVFACTOR 32768.0f

#define SOURCE 0

char *ID;
/* My process rank           */
int my_id;
/* The number of processes   */
int procs;

/* Matrixes given by a pointer */
float *A, *B, *X;



/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}


/* Set the program parameters from the command-line arguments */
/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int submit = 0;  /* = 1 if submission parameters should be used */
  int seed = 0;  /* Random seed */
  // char uid[L_cuserid + 2]; /*User name */

  if (argc == 3) {
    seed = atoi(argv[3]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  }
  else {
    printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
           argv[0]);
    printf("       %s submit\n", argv[0]);
    exit(0);
  }
  /* Interpret command-line args */
  if (!submit) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
    procs = atoi(argv[2]);
    if (procs < 1) {
      printf("Warning: Invalid number of processors = %i.  Using 1.\n", procs);
      procs = 1;
    }
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
  printf("Number of processors = %i.\n", procs);
}


/* Allocates memory for A, B and X */
void allocate_memory() {
    A = (float*)malloc( N*N*sizeof(float) );
    B = (float*)malloc( N*sizeof(float) );
    X = (float*)malloc( N*sizeof(float) );
}

/* Free allocated memory for arrays */
void free_memory() {
    free(A);
    free(B);
    free(X);
}

/* Print input matrices */
void print_inputs() {
  int row, col;
  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
    printf("%5.2f%s", A[col+N*row], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

/* Print matrix A */
void print_A() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
    printf("%5.2f%s", A[col+N*row], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

/* Print matrix B */
void print_B() {
  int col;
  if (N < 10) {
      printf("\nB = [");
        for (col = 0; col < N; col++) {
          printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
        }
    }
}
/* Print matrix X */
void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {

  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[col+N*row] = (float)rand() / DIVFACTOR;
    }
    B[col] = (float)rand() / DIVFACTOR;
    X[col] = 0.0;
  }
}

int main(int argc, char **argv) {
  ID = argv[argc-1];
  argc--;
  /* Prototype functions*/
  void gauss();
  void backSubstitution();

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  
  /* Process program parameters */
  parameters(argc, argv);
  
  /* Define the array as the pointer type, so we need to 
     allocate space for A, B, X */
  allocate_memory();

  if (my_id == 0) {
  /* Initialize Matrix A and B */
    initialize_inputs();
  /* Print Input Matrix */
    print_inputs();
  }

  gauss();

  if (my_id == 0) {
    backSubstitution();
    // end_wtime = MPI_Wtime();
    /* Display timing results */
    // printf("wall clock time = %f\n", (end_wtime - start_wtime) * 1000);

    /* Display output */
    print_A();
    print_B();
    print_X();
  }

  /* Free the storage that assign before */
  free_memory();

  /* Terminal MPI */
  MPI_Finalize();
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */

void gauss() {

  MPI_Status status;
  int norm, row, col, i;
  int numprocs = procs;
  float multiplier;
  
  double start_wtime = 0.0, end_wtime;
  // MPI_Barrier(MPI_COMM_WORLD);
  if (my_id == 0) {
    printf("\nComputing Parallely Using MPI.\n");
    start_wtime = MPI_Wtime();
  }
  /* Gaussian Elimination */
  for (norm = 0; norm < N - 1; norm++) {
    MPI_Bcast( &A[ N * norm ], N, MPI_FLOAT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD );
    if (my_id == 0) {
      /* Row assigned for processor 0 */
      row = norm + 1;
      /* Row assigned for other processors */
      int row_new;

      /* Use to represent the current processor ID 
         Start from 1 becasue processor 0 will do the part of itself */
      int counter_send = 1;

      /* assign tasks(send part of array) from processor 0 to other processors */
      for (row_new = row + counter_send; row_new < N; ++row_new) {
        /* Interleaving Assign, If current ID is the last processor, 
           initialize the counter */
        if (counter_send == numprocs) {
          counter_send = 1;
          continue;
        }
        /* Send tasks to each processors */
        MPI_Send(&A[N * row_new], N, MPI_FLOAT, counter_send, 0, MPI_COMM_WORLD);
        MPI_Send(&B[row_new], 1, MPI_FLOAT, counter_send, 1, MPI_COMM_WORLD);
        /* For next processor */
        ++counter_send;
      }

      /* Tasks for processor 0 */
      for (row = norm + 1; row < N; row += numprocs) {
        multiplier = A[N * row + norm] / A[N * norm + norm];
        for (col = norm; col < N; col++) {
          A[N * row + col] -= A[N * norm + col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
      }
      /* MPI_RECV(buf, count, datatype, source, tag, comm, status) */
      for (i = 1; i < procs; ++i) {
        for (row = norm + 1 + i; row < N; row += procs) {
          MPI_Recv(&A[N *row], N, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
          MPI_Recv(&B[row], 1, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &status);
        }
      }
      if (norm == N - 2) {
        end_wtime = MPI_Wtime();
        printf("elapsed time = %f\n", end_wtime - start_wtime);
      }
    } // if (my_id == 0)
    else {      
      /* Receive the tasks from processor 0 */
      for (row = norm + 1 + my_id; row < N; row += numprocs) {
        MPI_Recv(&A[N * row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);

        /* Do the job in each processor */
        multiplier = A[N * row + norm] / A[N * norm + norm];
        for (col = norm; col < N; ++col) {
          A[N * row + col] -= A[N * norm + col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;

        /* Send back the result to the processor 0 */
        MPI_Send(&A[N * row], N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&B[row], 1, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);     
  } // end norm
}

/* Back substitution */
void backSubstitution() {  
  int row, col;
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N - 1; col > row; col--) {
      X[row] -= A[N * row + col] * X[col];
    }
    X[row] /= A[N * row + row];
  }
}


