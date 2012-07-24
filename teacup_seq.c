/*
 * teacup_seq.c
 *
 *  Created on: Jul 22, 2012
 *      Author: Pasieka Manuel , mapa17@posgrado.upv.es
 */

#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <cblas.h>

#include "teacup_tools.h"

int main(int argc, char **argv)
{
	double *A, *B, *C;
	int N;
	int i, j;
	double elapsed;

	// Input data
	N = atoi(argv[1]);

	A = (double *) malloc( N * N * sizeof(double) );
	B = (double *) malloc( N * N * sizeof(double) );
	C = (double *) malloc( N * N * sizeof(double) );
	if((A == NULL) || (B == NULL) || (C == NULL) ){
	  printf("Running out of memory!\n"); exit(EXIT_FAILURE);
	}

	//Fill matrixes. Generate Identity like matrix for A and B , So C should result in an matrix with a single major diagonal
	for(i=0; i < N; i++ ){
	 for(j=0; j < N; j++){
		A[i+N*j] = (i==j)?i:0.0;
		B[i+N*j] = (i==j)?1.0:0.0;
		C[i+N*j] = 0.0;
	 }
	}

	int rows = N, columns = N;
	int stride = N;
	tick();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, columns, 1.0, A, stride, B, stride, 1.0, C, stride);
	elapsed = tack();

	printf("%f sec\n", elapsed);


	if( N < 30 )
	{
		printf("C ... \n");
		for (i=0; i<N; i++) {
		 for (j=0; j<N; j++) {
			 printf("%3.1f ", C[i+N*j]);
		 }
		 printf("\n");
		}
	}

	free(A);
	free(B);
	free(C);

	exit(EXIT_SUCCESS);
}


