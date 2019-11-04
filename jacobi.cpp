
#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <time.h>

#define ITERATION_LIMIT 5
#define EPSILON 0.0000001
#define NUM_OF_THREADS 4

void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side);
// frist point
void init_array_parallel(float array[], int array_size);
// copy last solo ..
float* clone_array_parallel(float array[], int array_length);





int main(){

    int matrix_size = 3;

    float** matrix = new float*[matrix_size];

    for (int i = 0; i < matrix_size; i++) {
        matrix[i] = new float[matrix_size];
    }

    float* right_hand_side = new float[matrix_size];

    matrix[0][0] = 2 ;matrix[0][1] = -1;matrix[0][2] =  0;
    matrix[1][0] = -1;matrix[1][1] = -2;matrix[1][2] = -1;
    matrix[2][0] = 0 ;matrix[2][1] = -1;matrix[2][2] =  2;

    right_hand_side[0] = 7;
    right_hand_side[1] = 1;
    right_hand_side[2] = 1;


    // Computing the time
    const clock_t parallel_starting_time = clock();

    omp_set_num_threads(NUM_OF_THREADS);

    solve_jacobi_parallel(matrix, matrix_size, right_hand_side);

    printf("Elapsed time: %f ms\n", float(clock() - parallel_starting_time));

    return 0;
}

void solve_jacobi_parallel(float** matrix, int matrix_size, float* right_hand_side) {

	float* solution = new float[matrix_size];
	float* last_iteration = new float[matrix_size];

	// Just for initialization ..

	init_array_parallel(solution, matrix_size); // dump the array with zeroes

	// NOTE: we don't need to parallelize this as the iterations are dependent. However, we may parallelize the inner processes
	for (int i = 0; i < ITERATION_LIMIT; i++){
		// Make a deep copy to a temp array to compare it with the resulted vector later
		last_iteration = clone_array_parallel(solution, matrix_size);

		// Each thread is assigned to a row to compute the corresponding solution element
		#pragma omp parallel for schedule(dynamic, 1)
		for (int j = 0; j < matrix_size; j++){
			float sigma_value = 0;
			for (int k = 0; k < matrix_size; k++){
				if (j != k) {
					sigma_value += matrix[j][k] * solution[k];
				}
			}
			solution[j] = (right_hand_side[j] - sigma_value) / matrix[j][j];
		}

		// Checking for the stopping condition ...
		int stopping_count = 0;
		#pragma omp parallel for schedule(dynamic, 1)
		for (int s = 0; s < matrix_size; s++) {
			if (abs(last_iteration[s] - solution[s]) <= EPSILON) {
				#pragma atomic
				stopping_count++;
			}
		}

		//if (stopping_count == matrix_size) break;

		printf("Iteration #%d: ", i+1);
		for (int l = 0; l < matrix_size; l++) {
			printf("%f ", solution[l]);
		}
		printf("\n");
	}
}

void init_array_parallel(float arrayb[], int array_size){
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_size; i++) {
		arrayb[i] = 0;
	}
}

float* clone_array_parallel(float arrayb[], int array_length){
	float* output = new float[array_length];
	#pragma omp parallel for schedule (dynamic, 1)
	for (int i = 0; i < array_length; i++) {
		output[i] = arrayb[i];
	}
	return output;
}

