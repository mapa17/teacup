/*
 * teacup_par.c
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
//#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#include "teacup_tools.h"

int mpi_size, mpi_id;


int main(int argc, char **argv)
{
	int N;
	int nThreads;
	int nColumns;
	int i,j,k;
	double *A,*Bi,*C,*Ci;
	int BiRows, BiColumns;
	CompressedMatrix *cBi;
	CompressedMatrix *cCi;
	double elapsed;

	char printDebug;

	//************ Check Input **************/
	if(argc < 3){
		printf("Usage: %s MaxtrixSize NumberOfThreads\n" , argv[0] );
		exit(EXIT_FAILURE);
	}

	N = atoi(argv[1]);
	if( N <= 1){
		printf("MatrixSize must be bigger than 1!");
		exit(EXIT_FAILURE);
	}

	nThreads = atoi(argv[2]);
	if( nThreads <= 1){
		printf("NumberOfThreads must be bigger than 1!");
		exit(EXIT_FAILURE);
	}

	omp_set_num_threads(nThreads);
	omp_set_schedule(omp_sched_dynamic, N/10);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	nColumns = N / mpi_size; //For the moment depend on N being a multiple the number of MPI nodes

	//************ Prepare Matrix **************/
	A = (double *) malloc( N*N * sizeof(double) );
	if((A == NULL) ){
	  printf("Running out of memory!\n"); exit(EXIT_FAILURE);
	}

//	if(mpi_id != 0){
//		MPI_Finalize();
//		exit(0);
//	}

	if(mpi_id == 0)
	{
		printDebug = 0;

		if(printDebug) printf("[%d] Generating A ...",mpi_id);
		//Fill matrixes. Generate Identity like matrix for A and B , So C should result in an matrix with a single major diagonal
		for(i=0; i < N; i++ ){
		 for(j=0; j < N; j++){
			A[i+N*j] = (i==j)?i:0.0;

//			//Sparse Matrix with 10% population
//			A[i+N*j] = rand()%10;
//			if(A[i+N*j] == 0)
//				A[i+N*j] = rand()%10;
//			else
//				A[i+N*j] = 0;
		 }
		}

//		printMatrix(A, N, nColumns);
//		cA = compressMatrix(A, N, nColumns);
//		printCompressedMatrix(cA);
//		uncompressMatrix(cA, &Bi, &i, &j);
//		printMatrix(Bi, i, j);
//
//		MPI_Finalize();
//		exit(0);

		tick();

		if(printDebug) printf("[%d] Broadcasting A ...",mpi_id);
		MPI_Bcast( A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if(printDebug) printf("[%d] Generating B ...",mpi_id);
		double* B; CompressedMatrix* cB;
		B = (double *) malloc( N*N * sizeof(double) );
		for(i=0; i < N; i++ ){
		 for(j=0; j < N; j++){
			B[j+N*i] = (i==j)?1.0:0.0;
		 }
		}

		if(printDebug) printf("[%d] Compressing and distributing Bi ...",mpi_id);
		cB = compressMatrix(B, N, N);
		for(i=1; i < mpi_size; i++){
			mpiSendCompressedMatrix(cB, i*nColumns, (i+1)*nColumns, i);
		}

		//Fake shorten cB
		free(B);
		cB->columns = nColumns;
		uncompressMatrix(cB, &Bi, &BiRows, &BiColumns);
		Ci = MatrixMultiply(A, N, N, Bi, nColumns);

		if(printDebug) printf("[%d] Ci = A x Bi ...", mpi_id);
		if(printDebug) printMatrix(Ci, N, nColumns);

		cCi = compressMatrix(Ci, N, nColumns);
		if(printDebug) printf("cCi ...\n");
		if(printDebug) printCompressedMatrix(cCi);

		MPI_Barrier(MPI_COMM_WORLD);

		if(printDebug) printf("[%d] Receiving Ci fragments ...\n", mpi_id);
		CompressedMatrix** Cii;
		Cii = (CompressedMatrix**) malloc(sizeof(CompressedMatrix*) * mpi_size);
			if(Cii == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
		Cii[0] = cCi;
		for(i=1; i < mpi_size; i++){
			Cii[i] = mpiRecvCompressedMatrix(N,nColumns, i);
		}

		if(printDebug) printf("[%d] Joining Cii ...\n", mpi_id);
		CompressedMatrix *cC;
		cC = joinCompressedMatrices(Cii, mpi_size);
		if(printDebug) printCompressedMatrix(cC);

		elapsed =  tack();

		printf("[%d] C ...\n", mpi_id);
		uncompressMatrix(cC, &C, &i,&j);
		if(i <= 20){
			printMatrix(C, i,j);
		} else {
			if(i < 1000){
				printf("C is too big, only printing first diagonal %d.\n[",j);
				for(k=0; (k < i) && (k < j); k++){
					printf("%3.2f ",C[k + k*j]);
				}
				printf("]\n");
			} else {
				printf("C is just too big!");
			}
		}

		printf("Took [%f] seconds!\n",elapsed);

	} else {
		printDebug = 0;

		if(printDebug) printf("[%d] Waiting for A ...",mpi_id);
		MPI_Bcast( A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if(printDebug) printf("[%d] Received A ...\n", mpi_id);
		if(printDebug) printMatrix(A, N, N);

		if(printDebug) printf("[%d] Waiting for Bi ...",mpi_id);
		cBi = mpiRecvCompressedMatrix(N, nColumns, 0);
		uncompressMatrix(cBi, &Bi, &BiRows, &BiColumns);

		if(printDebug) printf("[%d] Received Bi ...",mpi_id);
		if(printDebug) printMatrix(Bi,BiRows, BiColumns);

		assert( (BiRows == N) && "Number or Rows in Bi is not right!");
		assert( (BiColumns == nColumns) && "Number or Columns in Bi is not right!");

		Ci = MatrixMultiply(A, N, N, Bi, BiColumns);

		if(printDebug) printf("[%d] Ci = A x Bi ...", mpi_id);
		if(printDebug) printMatrix(Ci, N, nColumns);

		cCi = compressMatrix(Ci, N, nColumns);
		if(printDebug) printCompressedMatrix(cCi);

		MPI_Barrier(MPI_COMM_WORLD);

		if(printDebug) printf("[%d] Returning Ci ...\n", mpi_id);
		mpiSendCompressedMatrix(cCi, 0, nColumns, 0);

	}


	MPI_Finalize();
	// NxM = NxN * NxM
	exit(EXIT_SUCCESS);
}
