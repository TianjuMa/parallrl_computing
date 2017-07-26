/*Gaussian elimination without pivoting */

/* Algorithm description:                                                                  
/*                                                                                         
/*   In the serial algorithm, there are three loops. We do the parallel at the second loop by
     consider the efficiency and feasibility.

     The number of rows that every thread processes changes in every iteration because 
     the number of rows to apply the factor dimishes.

     For dynamic schedule, the tasks will not be totally assign to proccessors in one time.
*/

/* Tianju Ma, Yuhang Peng */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>

/* To program a parallel application with Pthreads, we need this source file */
/* Sild 7.20 */
#include <pthread.h>

/*#include <ulocks.h>
#include <task.h>
*/

char *ID;

/* Program Parameters */
#define MAXN 6000  /* Max value of N */
int N;  /* Matrix size */
int procs;  /* Number of processors to use */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
                * It is this routine that is timed.
                * It is called only on the parent.
                */

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int submit = 0;  /* = 1 if submission parameters should be used */
  int seed = 0;  /* Random seed */
  char uid[L_cuserid + 2]; /*User name */

  /* Read command-line arguments */
  //  if (argc != 3) {
  if ( argc == 1 && !strcmp(argv[1], "submit") ) {
    /* Use submission parameters */
    submit = 1;
    N = 4;
    procs = 2;
    printf("\nSubmission run for \"%s\".\n", cuserid(uid));
      /*uid = ID;*/
    strcpy(uid,ID);
    srand(randm());
  }
  else {
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
  }
    //  }
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

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }
}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
        printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 10) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  ID = argv[argc-1];
  argc--;

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();

  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
         (float)(usecstop - usecstart)/(float)1000);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, procs, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */

/* Chunk size for Dynamic shedule */
#define CHUNK 1

/* Lock for global variable inside thread function */
pthread_mutex_t gauss_lock;

/* Barrier variable */
pthread_mutex_t barrier_mutex;
pthread_cond_t go;
int numArrived = 0;

/* The row for each norm */
int current_row = 0;

/* Thread function, parallel computing inside */
void barrier(){

  pthread_mutex_lock(&barrier_mutex);
  numArrived++;

  /* If the number of threads arrive at barrier() less than the total
     numebr of worker threads, wait*/
  if (numArrived<procs) {
    pthread_cond_wait(&go, &barrier_mutex);
  }
  /* All threads arrival at the barrier(), remove barrier, and initialize 
     the numArrived for preparing for next barrier */
  else {
    pthread_cond_broadcast(&go);
    numArrived=0; 
  }
  pthread_mutex_unlock(&barrier_mutex);
}

void *gaussian_elimination(void *s){
  int threadid;
  threadid = *((int*)(&s));
  
  /* Due to dynamic schedule, each thread won't be assign the tasks at once */

  /* Set current norm for Dynamic sheduling*/
  int current_norm = 0; 
  
  /* Local_row for each process */
  int local_row;

  /* Gaussian Elimination */
  float multiplier;
  for (current_norm = 0; current_norm < N; ++current_norm) {
    /* For each norm, we need to change the row for starting */
    current_row = current_norm + 1;
    /* Parallel computing */
    for (current_row; current_row < N; ) {
      /* Set the lock for updating global variable */
      pthread_mutex_lock(&gauss_lock);
        local_row = current_row;
        current_row += CHUNK;
      pthread_mutex_unlock(&gauss_lock);
      /* Computing part */
      multiplier = A[local_row][current_norm] / A[current_norm][current_norm];
      for (int col = current_norm; col < N; ++col){
        A[local_row][col] -= A[current_norm][col] * multiplier;
      }
      B[local_row] -= B[current_norm] * multiplier;
    }
    /* Set barrier to wait current norm finished */
    barrier();
  }
}

/* Entrance of gauss function */
void gauss(){
  int norm, row, col;
  printf("Computing in pthread!\n");

  pthread_t threads[procs];

  /* Initialize the Barrier */
  pthread_mutex_init(&barrier_mutex, NULL);
  pthread_cond_init(&go, NULL);

  /* Initialize the Thread */
  pthread_mutex_init(&gauss_lock, NULL);
  for (int i = 0; i < procs; ++i){
    pthread_create(&threads[i], NULL, &gaussian_elimination, (void*)i);
  }
  for (int i = 0; i < procs; ++i) {
    pthread_join(threads[i], NULL);
  }

  /* Back substitution */
  for (row = N - 1; row >= 0; --row) {
    X[row] = B[row];
    for (col = N-1; col > row; --col) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}