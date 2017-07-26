#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_POINTS 524288

unsigned int X_axis[NUM_POINTS];
unsigned int Y_axis[NUM_POINTS];

//swao the array elemnt at index i and j
void swap(unsigned int array[], int i, int j);
int min(int x, int y);
int max(int x, int y);
void trim(unsigned int *X_axis, unsigned int *Y_axis, int x_pivot, int points);
//find the kth smallest element in array v
unsigned int find_kth(unsigned int *v, int n, int k, unsigned int *y);
//loop indeces
int i = 0;
int j = 0;
int k = 0;
int z = 0;
//global_cost computed at process 0
double global_cost = 0;

int numprocs;  /* Number of processors to use */
int myid;

int num_quadrants;
void find_quadrants (num_quadrants)
{  
   if (myid == 0) {
     //x_cut = 0 means it is time to cut at x dimension, 1 in y dimension
    int x_cut = 0;
    //record the number of quadrants right now
    int quadrants = 1;

    //coordinates of the quadrants, top and bottom means the smallest and largest y axis;
    //left and right means the smallest and largest y axis.
    int top[num_quadrants];
    int left[num_quadrants];
    int right[num_quadrants];
    int bottom[num_quadrants];
 
    //record the bisection coordinates
    int pivot_arr[num_quadrants];
    //the loop ends when there are enough quadrants
    while (num_quadrants > quadrants) { 
        //the points in each current quadrant
        int points = NUM_POINTS / quadrants;
        //bisection on X dimension
        if (!x_cut) {
            for (i = 0; i < quadrants; i++) {
              //find the median of the corresponding part of X_axis[]
                int x_pivot = find_kth(X_axis + i * points, points, points/2 - 1, Y_axis + i * points);
                trim(X_axis, Y_axis, x_pivot, points);
                //recond the dividing point
                pivot_arr[i+quadrants-1] = x_pivot;
            }
            //change to Y's turn
            x_cut = 1;  
        } else {
          //bisection on Y dimension 
            for (i = 0; i < quadrants; i++) {
                int y_pivot = find_kth(Y_axis + i * points, points, points/2 - 1, X_axis + i * points);
                trim(Y_axis, X_axis, y_pivot, points);
                //recond the dividing point
                pivot_arr[i+quadrants-1] = y_pivot;
            }
            //change the X'turn
            x_cut = 0;
        }
        //After one's turn, the current number of quadrants is doubled
        quadrants *= 2;

    }

    //find border of initial quadrant by finding the border of X and Y
    int min_x = X_axis[0];
    int max_x = X_axis[0];
    int min_y = Y_axis[0];
    int max_y = Y_axis[0];
    for (i = 1; i < NUM_POINTS; i++) {
        min_x = min(X_axis[i], min_x);
        max_x = max(X_axis[i], max_x);
        min_y = min(Y_axis[i], min_y);
        max_y = max(Y_axis[i], max_y);
    }
    //update the quadrants' coordinates according to the values in the pivot_array[]
    top[0] = min_y;
    bottom[0] = max_y;
    left[0] = min_x;
    right[0] = max_x;
    x_cut = 0;
    quadrants = 1;
    //i is the number of quadrants computed so far

    for (i = 0; i < num_quadrants-1; i += quadrants) {
        int temp = i;
        //copy the coordinates to the last half of the arrays 
        for (j = 0; j < quadrants; j++) {
            top[j+quadrants] = top[j];
            bottom[j+quadrants] = bottom[j];
            left[j+quadrants] = left[j];
            right[j+quadrants] = right[j];
        }
        //X's turn
        if (!x_cut) {
          //divide the quadrant into two halves, the only coordinates changed are the right of the left subquadrant and the left of the right subquadrant
            for (j = 0; j < quadrants * 2; j+=2) {
                top[j] = top[quadrants + j/2];
                bottom[j] = bottom[quadrants + j/2];
                left[j] = left[quadrants + j/2];
                right[j] = pivot_arr[temp];

                top[j+1] = top[quadrants + j/2];
                bottom[j+1] = bottom[quadrants + j/2];
                left[j+1] = pivot_arr[temp];
                right[j+1] = right[quadrants + j/2];
                temp++;
            }
            x_cut = 1;
        } else{
          //Y's turn
          //divide the quadrant into two halves, the only coordinates changed are the bottom of the upper subquadrant and the top of the lower subquadrant
            for (j = 0; j < quadrants * 2; j+=2) {
                top[j] = top[quadrants + j/2];
                bottom[j] = pivot_arr[temp];
                left[j] = left[quadrants + j/2];
                right[j] = right[quadrants + j/2];
                
                top[j+1] = pivot_arr[temp];
                bottom[j+1] = bottom[quadrants + j/2];
                left[j+1] = left[quadrants + j/2];
                right[j+1] = right[quadrants + j/2];
                temp++;
            }
            x_cut = 0;
        }
        quadrants *= 2;
    }
    }
   //broadcast the X and Y from 0 to all other proesses
    MPI_Bcast(&X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);  

    double local_cost = 0;
    for (i = myid; i < num_quadrants; i += numprocs)
    {
       int points = NUM_POINTS / num_quadrants;
       for (j = 0; j < points - 1; j++) {
            for (k = j+1; k < points; k++) {
                int x1 = points * i + j;
                int x2 = points * i + k;
                int y1 = points * i + j;
                int y2 = points * i + k;

                double diff_x = abs(X_axis[x1] - X_axis[x2]);
                double diff_y = abs(Y_axis[y1] - Y_axis[y2]);
                local_cost += sqrt((double)diff_x * diff_x + diff_y * diff_y);
            }
       }     
    }
    //reduce and calculate the global_cost on process 0
     MPI_Reduce(&local_cost, &global_cost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

//find the kth smallest key in v
unsigned int find_kth(unsigned int *v, int n, int k, unsigned int *y) {
    int j0 = 0;
    int i1 = 0;
    int j1 = 0;
    if (n == 1 && k == 0) return v[0];
    //divide the array into n/5 subarrays of 5 elements and find the medians for each on the subarrays
    int m = (n + 4)/5;
    unsigned int *medians =  (unsigned int *)malloc(m * sizeof(int));
    for (i1=0; i1<m; i1++) {
        if (5*i1 + 4 < n) {
            unsigned int *w = v + 5*i1;
            unsigned int *w1 = y + 5*i1;
            for (j0=0; j0<3; j0++) {
                int jmin = j0;
                for (j1=j0+1; j1<5; j1++) {
                    if (w[j1] < w[jmin]) jmin = j1;
                }
                swap(w, j0, jmin);
                swap(w1, j0, jmin);
            }
            medians[i1] = w[2];
        } else {
            medians[i1] = v[5*i1];
        }
    }
    //find the median of medians
    int pivot = find_kth(medians, m, m/2, medians);
    free(medians);
 
    for (i1=0; i1<n; i1++) {
        if (v[i1] == pivot) {
            swap(v, i1, n-1);
            swap(y, i1, n-1);
            break;
        }
    }
 
    int store = 0;
    for (i1=0; i1<n-1; i1++) {
        if (v[i1] < pivot) {
            swap(v, i1, store);
            swap(y, i1, store);
            store++;
        }
    }
    swap(v, store, n-1);
    swap(y, store, n-1);
    
    if (store == k) {
       //return when exactly k elemens are smaller than pivot
        return pivot;
    } else if (store > k) {
      //find kth smallest in the left store amount of elements in v
        return find_kth(v, store, k, y);
    } else {
      //find the k-store-1 smallest in the right side of the v staring at index v+store+1
        return find_kth(v+store+1, n-store-1, k-store-1, y+store+1);
    }
}


void swap(unsigned int array[], int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

int min(int x, int y) {
    return x < y ? x : y;
}

int max(int x, int y) {
    return x < y ? y : x;
}
void trim(unsigned int *X_axis, unsigned int *Y_axis, int x_pivot, int points) {
    k = i * points;
    j = (i + 1) * points - 1;
    while (k < j && k < k + points/2) {
        if (X_axis[k] > x_pivot) {
            while (X_axis[j] > x_pivot) {
                j--;
            }
            if (k < j) {
                swap(X_axis, k, j);
                swap(Y_axis, k, j);
            } 
        }
        k++;
    }
}
int main(argc,argv)
int argc;
char *argv[];
{
    int num_quadrants;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    /*Time Variables*/
    double startwtime = 0.0, endwtime;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);
    
    if (argc != 2)
    {
        fprintf (stderr, "Usage: recursive_bisection <#of quadrants>\n");
        MPI_Finalize();
        exit (0);
    }
    
    fprintf (stderr,"Process %d on %s\n", myid, processor_name);
    
    num_quadrants = atoi (argv[1]);
    
    if (myid == 0)
        fprintf (stdout, "Extracting %d quadrants with %d processors \n", num_quadrants, numprocs);
        
        if (myid == 0)
        {
            int i;
            
            srand (10000);
            
            for (i = 0; i < NUM_POINTS; i++)
                X_axis[i] = (unsigned int)rand();

            
            for (i = 0; i < NUM_POINTS; i++)
                Y_axis[i] = (unsigned int)rand();

            //start timer at process 0
            printf("\nComputing Parallely Using MPI.\n");
            startwtime = MPI_Wtime();
        }
    
    MPI_Bcast(&X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);  
    
    find_quadrants (num_quadrants);
  //barrier to guarantee the correctness of timing
     MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0) {
      //end timer at process 0
        endwtime = MPI_Wtime();
        printf("\nelapsed time = %f\n", endwtime - startwtime);
        printf("\nTotal cost:  %lf \n", global_cost);

    }
    
    MPI_Finalize();


    return 0;
}


