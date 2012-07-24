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
#include <cblas.h>
#include <mpi.h>

#include "teacup_tools.h"

int mpi_size, mpi_id;

typedef struct {
	int rows;
	int columns;
	int* nNoneZero;
	double** NoneZero;
	int* nZero;
	int** Zero;
}CompressedMatrix;

void printCompressedMatrix(CompressedMatrix* cM);
CompressedMatrix* compressMatrix(double* M, int nRows, int nColumns);
void uncompressMatrix(CompressedMatrix* cM, double **Matrix, int* nRows, int *nColumns);
void mpiSendCompressedMatrix(CompressedMatrix* cM, int startColumn, int endColumn, int destID);
CompressedMatrix* mpiRecvCompressedMatrix(int N, int nColumns, int srcID);
double* MatrixMultiply(double* A, int nRowsA, int n, double* B, int nColumnsB);
CompressedMatrix* joinCompressedMatrices(CompressedMatrix **Cii, int nMatrices);
//CompressedMatrix* extractCompressedMatrix(CompressedMatrix* M, int columnStart, int columnEnd);

int main(int argc, char **argv)
{
	int N;
	int nThreads;
	int nColumns;
	int i,j,k;
	double *A,*Bi,*C,*Ci;
	int BiRows, BiColumns;
	CompressedMatrix *cBi,*cA;
	CompressedMatrix *cCi;

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

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	nColumns = N / mpi_size; //For the moment depend on N being a multiple the number of MPI nodes

	//************ Prepare Matrix **************/
	A = (double *) malloc( N*N * sizeof(double) );

//	Ci = (double *) malloc( N*nColumns * sizeof(double) );
	if((A == NULL) ){
	  printf("Running out of memory!\n"); exit(EXIT_FAILURE);
	}

//	printf("Node [%d] Bi ...",mpi_id);
//	printMatrix(Bi, N, nColumns);



	if(mpi_id == 0)
	{
		printf("[%d] Generating A ...",mpi_id);
		//Fill matrixes. Generate Identity like matrix for A and B , So C should result in an matrix with a single major diagonal
		for(i=0; i < N; i++ ){
		 for(j=0; j < N; j++){
			A[i+N*j] = (i==j)?i:0.0;

			//Sparse Matrix with 10% population
	//		A[i+N*j] = rand()%10;
	//		if(A[i+N*j] == 0)
	//			A[i+N*j] = rand()%10;
	//		else
	//			A[i+N*j] = 0;
		 }
		}

		printf("[%d] Broadcasting A ...",mpi_id);
		MPI_Bcast( A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		printf("[%d] Generating B ...",mpi_id);
		double* B; CompressedMatrix* cB;
		B = (double *) malloc( N*N * sizeof(double) );
		for(i=0; i < N; i++ ){
		 for(j=0; j < N; j++){
			B[j+N*i] = (i==j)?1.0:0.0;
		 }
		}

		printf("[%d] Compressing and distributing Bi ...",mpi_id);
		cB = compressMatrix(B, N, N);
		for(i=1; i < mpi_size; i++){
			mpiSendCompressedMatrix(cB, i*nColumns, (i+1)*nColumns, i);
		}

		//Fake shorten cB
		free(B);
		cB->columns = nColumns;
		uncompressMatrix(cB, &Bi, &BiRows, &BiColumns);
		Ci = MatrixMultiply(A, N, N, Bi, nColumns);

		printf("[%d] Ci = A x Bi ...", mpi_id);
		printMatrix(Ci, N, nColumns);

		cCi = compressMatrix(Ci, N, nColumns);

		MPI_Barrier(MPI_COMM_WORLD);

		printf("[%d] Receiving Ci fragments ...\n", mpi_id);
		CompressedMatrix** Cii;
		Cii = (CompressedMatrix**) malloc(sizeof(CompressedMatrix*) * mpi_size);
			if(Cii == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
		Cii[0] = cCi;
		for(i=1; i < mpi_size; i++){
			printf("C%d ", i);
			Cii[i] = mpiRecvCompressedMatrix(N,nColumns, i);
		}

		printf("[%d] Joining Cii ...\n", mpi_id);
		CompressedMatrix *C;
		C = joinCompressedMatrices(Cii, mpi_size);


		printf("[%d] Cii ...\n", mpi_id);
		printCompressedMatrix(C);

	} else {
		printf("[%d] Waiting for A ...",mpi_id);
		MPI_Bcast( A, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		printMatrix(A, N, N);

		printf("[%d] Waiting for Bi ...",mpi_id);
		cBi = mpiRecvCompressedMatrix(N, nColumns, 0);
		uncompressMatrix(cBi, &Bi, &BiRows, &BiColumns);

		printf("[%d] Received Bi ...",mpi_id);
		printMatrix(Bi,BiRows, BiColumns);

		assert( (BiRows == N) && "Number or Rows in Bi is not right!");
		assert( (BiColumns == nColumns) && "Number or Columns in Bi is not right!");

		Ci = MatrixMultiply(A, N, N, Bi, BiColumns);

		printf("[%d] Ci = A x Bi ...", mpi_id);
		printMatrix(Ci, N, nColumns);

		cCi = compressMatrix(Ci, N, nColumns);

		MPI_Barrier(MPI_COMM_WORLD);

		printf("[%d] Returning Ci ...\n", mpi_id);
		mpiSendCompressedMatrix(cCi, 0, nColumns, 0);

	}


	MPI_Finalize();
	// NxM = NxN * NxM
	exit(EXIT_SUCCESS);
}

//CompressedMatrix* extractCompressedMatrix(CompressedMatrix* M, int columnStart, int columnEnd)
//{
//	CompressedMatrix *C;
//	double* t; int* t2;
//	int nTotalNZ, nTotalZ;
//	int i,j;
//
//	assert( (columnStart>0) && (columnStart < M->columns) && "Invalid index");
//	assert( (columnEnd > columnStart) && (columnEnd <= M->columns) && "Invalid index");
//
//	C->columns = columnEnd - columnStart;
//	C->rows = M->rows;
//
//	C->nNoneZero = malloc(sizeof(int)*C->columns);
//		if(C->nNoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
//	C->NoneZero = malloc(sizeof(double*)*C->columns);
//		if(C->NoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
//	C->nZero = malloc(sizeof(int)*C->columns);
//		if(C->nZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
//	C->Zero = malloc(sizeof(int*)*C->columns);
//		if(C->Zero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
//
//
//
//
//	return C;
//}

CompressedMatrix* joinCompressedMatrices(CompressedMatrix **Cii, int nMatrices)
{
	CompressedMatrix *C;
	int i,j,k;

	int nTotalColumns, nRows;
	nRows = Cii[0]->rows;

	nTotalColumns = 0;
	for(i = 0; i<nMatrices; i++){
		nTotalColumns += Cii[i]->columns;
		assert( (nRows == Cii[i]->rows) && "The number of rows in each matrix must be the same!" );
	}

	C = malloc(sizeof(CompressedMatrix));
		if(C == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	C->rows = nRows;
	C->columns = nTotalColumns;
	C->nNoneZero = malloc(sizeof(int)*C->columns);
		if(C->nNoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	C->NoneZero = malloc(sizeof(double*)*C->columns);
		if(C->NoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	C->nZero = malloc(sizeof(int)*C->columns);
		if(C->nZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	C->Zero = malloc(sizeof(int*)*C->columns);
		if(C->Zero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	k = 0;
	for(i = 0; i < nMatrices; i++){
		for(j = 0; i < Cii[i]->columns; i++){
			C->nNoneZero[k+j] = Cii[i]->nNoneZero[j];
			C->NoneZero[k+j] = Cii[i]->NoneZero[j];
			C->nZero[k+j] = Cii[i]->nZero[j];
			C->Zero[k+j] = Cii[i]->Zero[j];
		}
		k+=j;

		assert( (k <= nTotalColumns) && "Too many columns!");
	}

	return C;
}

printMatrix(double *M, int rows, int columns)
{
	int i,j;

	printf("\n");
	for(i = 0; i < rows; i++)
	{
		printf("[ ");
		for(j = 0; j < columns; j++)
		{
			printf("%3.1f ", M[j + i*columns]);
		}
		printf(" ]\n");
	}
	printf("\n");
}

CompressedMatrix* compressMatrix(double* M, int nRows,int nColumns)
{
	CompressedMatrix *cM;
	int **Z;
	double **NZ;
	int *tmp;
	double *tmp2;
	int i,j;
	int start, end;

	//printf("Reserve initial structure");

	cM = malloc(sizeof(CompressedMatrix));
		if(cM == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->nNoneZero = calloc(sizeof(int),nColumns);
		if(cM->nNoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->nZero = calloc(sizeof(int),nColumns);
		if(cM->nZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->rows = nRows;
	cM->columns = nColumns;

	//Reserve a lot of mem for worst case and than reduce later
	tmp = malloc(sizeof(int) * (2*cM->columns)*cM->rows);
		if(tmp == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	Z = (int**)malloc(sizeof(int*) * cM->columns);
		if(Z == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	tmp2 = malloc(sizeof(double) * cM->rows*cM->columns);
		if(tmp == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	NZ = (double**)malloc(sizeof(double*) * cM->columns);
		if(Z == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	for(i=0; i < cM->columns; i++){
		Z[i] = &tmp[ i*(2*cM->columns) ];
		NZ[i] = &tmp2[i*cM->columns];
	}

	//printf("Find Blocks");

	//Now create table of Zero blocks for each column!
	// i ... line, j ... column
	for(j=0; j<cM->columns; j++){
		for(i=0; i<cM->rows; i++){
			if(M[j + i*cM->rows] == 0.0){
				start = i;
				end = i+1;
				while( (end < cM->rows) && (M[j + end*cM->rows] == 0.0) ) end++;
				Z[j][cM->nZero[j]*2 + 0] = start;
				Z[j][cM->nZero[j]*2 + 1] = end;
				cM->nZero[j]++;
				i=end-1;
			} else {
				NZ[j][cM->nNoneZero[j]] = M[j + i*cM->rows];
				cM->nNoneZero[j]++;
			}
		}
	}

	//printf("Initialize final structure");
	//Now generate the final data structure
	int nTotalZeros, nTotalNoneZeros;
	int *t; double *t2; int pos; int pos2;
	nTotalNoneZeros = 0; nTotalZeros = 0;
	for(i = 0; i < cM->columns; i++){
		nTotalZeros += cM->nZero[i];
		nTotalNoneZeros += cM->nNoneZero[i];
	}
	t = malloc(sizeof(int) * nTotalZeros*2);
		if(t == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->Zero = (int**)malloc(sizeof(int*) * cM->columns);
		if(cM->Zero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	t2 = malloc(sizeof(double) * nTotalNoneZeros);
		if(t2 == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->NoneZero = (double**)malloc(sizeof(double*) * cM->columns);
		if(cM->NoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	//printf("Copy data");
	pos = 0; pos2 = 0;
	for(i=0; i < cM->columns; i++){
		cM->Zero[i] = &t[ pos ];
		pos += cM->nZero[i]*2;
		memcpy( cM->Zero[i], Z[i], sizeof(int)*2*cM->nZero[i]);

		cM->NoneZero[i] = &t2[ pos2 ];
		pos2 += cM->nNoneZero[i];
		memcpy( cM->NoneZero[i], NZ[i], sizeof(double)*cM->nNoneZero[i]);
	}

	free(tmp);
	free(tmp2);

	//printf("Finished\n");

	//printCompressedMatrix(cM);

	return cM;
}

void uncompressMatrix(CompressedMatrix* cM, double **Matrix, int* nRows, int *nColumns)
{
	int i,j,k;
	double *M;
	int nZ, nNZ;
	int rowPos;

	assert( (cM != NULL) && "Illegal reference!");
	assert( (nRows != NULL) && "Illegal reference" );
	assert( (nColumns != NULL) && "Illegal reference" );

	printf("Uncompressing Matrix [%d x %d] ...",cM->rows, cM->columns);

	*nRows = cM->rows; *nColumns = cM->columns;
	M = (double*)malloc(sizeof(double) * cM->rows*cM->columns);
		if(M == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	*Matrix = M;


	for(j=0; j < cM->columns; j++){
		nNZ = 0;
		rowPos = 0;
		k = 0;
		//printf("Column[%d] ",j);

		for(i=0; i < cM->nZero[j]; i++){
			//Fill NZ until start of ZeroBlock
			while(rowPos < cM->Zero[j][i*2+0]){
				M[j + rowPos*cM->columns] = cM->NoneZero[j][nNZ];
				nNZ++;
				rowPos++;
			}
			//Fill Zero block
			while(rowPos < cM->Zero[j][i*2+1]){
				M[j + rowPos*cM->columns] = 0.0;
				rowPos++;
			}
			//printf(" flip[%d, %d] ", rowPos, nNZ);
		}

		//printf(" flop[%d, %d] ", rowPos, nNZ);
		assert( (nNZ <= cM->nNoneZero[j]) && "Too many NZ values!");

		//Fill NZ until end of column
		while(nNZ < cM->nNoneZero[j]){
			M[j + rowPos*cM->columns] = cM->NoneZero[j][nNZ];
			rowPos++;
			nNZ++;
		}

		assert( (rowPos == cM->rows) && "Not correct row Position!");

		//printf("end\n");
	}

	printf("Finished!\n");
}

void printCompressedMatrix(CompressedMatrix* cM)
{
	int i,j;
	for(i=0; i<cM->columns; i++){
		printf("C[%d] nNZ[%d] NZ[",i, cM->nNoneZero[i]);
		for(j=0;j<cM->nNoneZero[i];j++){
			printf("%3.1f ",cM->NoneZero[i][j]);
		}
		printf("] nZ[%d] Z[",cM->nZero[i]);
		for(j=0;j<cM->nZero[i];j++){
			printf("(%d,%d) ",cM->Zero[i][j*2],cM->Zero[i][j*2+1]);
		}
		printf("]\n");
	}
}

void mpiSendCompressedMatrix(CompressedMatrix* cM, int startColumn, int endColumn, int destID)
{
	int i,j;
	int position;
	char* b;
	int bSize;

	int nTotalNoneZero = 0;
	int nTotalZero = 0;

	assert( (startColumn >= 0) && (startColumn < endColumn) && "Illegal column specified!" );
	assert( (endColumn <= cM->columns) && (startColumn < endColumn) && "Illegal column specified!" );

	printf("Sending compressed matrix, columns [%d-%d] to node [%d] ...\n",startColumn, endColumn, destID);

	for(i = startColumn; i < endColumn; i++)
	{
		nTotalNoneZero += cM->nNoneZero[i];
		nTotalZero += cM->nZero[i];
	}

	//Be sure to have enough buffer
	bSize = sizeof(int)*(nTotalZero*2 + 2*(endColumn-startColumn)) + sizeof(double) * ( nTotalNoneZero );
	b = malloc( bSize );

	printf("Reserving buffer of size [%d] for nTotalNZ[%d], nTotalZ[%d]\nPacking ...\n", bSize, nTotalNoneZero, nTotalZero);

	position=0;
	MPI_Pack(&(cM->nNoneZero[startColumn]), endColumn-startColumn, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->NoneZero[startColumn][0]), nTotalNoneZero, MPI_DOUBLE, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->nZero[startColumn]), endColumn-startColumn, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->Zero[startColumn][0]), nTotalZero*2, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );

	printf("Pack position [%d]\nSending ...",position);

	assert( (position == bSize) && "Something went wrong during packing!" );

	MPI_Send(&position, 1, MPI_INT, destID, 0, MPI_COMM_WORLD);
	MPI_Send(b, position, MPI_PACKED, destID, 0, MPI_COMM_WORLD);
	printf("Finished sending compressed Matrix!\n");
}

CompressedMatrix* mpiRecvCompressedMatrix(int nRows, int nColumns, int srcID)
{
	CompressedMatrix* cM;
	char *b;
	int bSize;
	int position;
	int nTotalNoneZero, nTotalZero;
	int i;
	double *b2; int *b3;

	assert( (nRows > 0) && "Illegal Matrix size!");
	assert( (nColumns > 0) && "Illegal number of columns to receive!");

	printf("Receiving compressed matrix [%d x %d] from node [%d] ...\n",nColumns, nRows, srcID);

	cM = malloc(sizeof(CompressedMatrix));
		if(cM == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->rows = nRows;
	cM->columns = nColumns;
	cM->nNoneZero = malloc(sizeof(int) * cM->rows);
		if(cM->nNoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	cM->nZero = malloc(sizeof(int) * cM->rows);
		if(cM->nZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	//Receive the size of the packed message and reserve a buffer of the same size
	MPI_Recv(&bSize, 1, MPI_INT, srcID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	assert( (bSize > 0) && ("Size for receive buffer must be bigger than zero!") );
	b = malloc(bSize);
		if(b == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	printf("Will need a receive buffer of size [%d]\n", bSize);

	//Receive the packed message and unpack it
	MPI_Recv(b, bSize, MPI_PACKED, srcID, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	position = 0;
	MPI_Unpack(b, bSize, &position, cM->nNoneZero, nColumns, MPI_INT, MPI_COMM_WORLD);
	nTotalNoneZero = 0;
	for(i=0; i<nColumns; i++){
		nTotalNoneZero += cM->nNoneZero[i];
	}
	b2 = malloc(sizeof(double) * nTotalNoneZero);
		if(b2 == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	MPI_Unpack(b, bSize, &position, b2, nTotalNoneZero, MPI_DOUBLE, MPI_COMM_WORLD);
	cM->NoneZero = malloc(sizeof(double*) * cM->columns);
		if(cM->NoneZero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	nTotalNoneZero = 0;
	for(i = 0; i < cM->columns; i++){
		cM->NoneZero[i] = &b2[nTotalNoneZero];
		nTotalNoneZero += cM->nNoneZero[i];
	}

	MPI_Unpack(b, bSize, &position, cM->nZero, nColumns, MPI_INT, MPI_COMM_WORLD);
	nTotalZero = 0;
	for(i=0; i<nColumns; i++){
		nTotalZero += cM->nZero[i];
	}
	b3 = malloc(sizeof(int) * nTotalZero*2);
		if(b3 == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	MPI_Unpack(b, bSize, &position, b3, nTotalZero*2, MPI_INT, MPI_COMM_WORLD);
	cM->Zero = malloc(sizeof(int*) * nTotalZero);
		if(cM->Zero == NULL){ perror("malloc"); exit(EXIT_FAILURE); }
	nTotalZero = 0;
	for(i = 0; i < cM->columns; i++){
		cM->Zero[i] = &b3[nTotalZero];
		nTotalZero += cM->nZero[i]*2;
	}


	printf("Finished unpacking at position [%d]\n", position);

	assert((position == bSize) && "Something went wrong during unpacking.");

	free(b);

	return cM;
}

//C = A * B , with A = [nRowsA x n] and B [ n x nColumnsB] , C = [nRowsA x nColumnsB]
double* MatrixMultiply(double* A, int nRowsA, int n, double* B, int nColumnsB)
{
	double *C;
	//C = malloc(sizeof(double) * nRowsA * nColumnsB);
	C = calloc(sizeof(double),nRowsA * nColumnsB);
		if(C == NULL){ perror("malloc"); exit(EXIT_FAILURE); }

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nRowsA, nColumnsB, n, 1.0, A, n, B, nColumnsB, 1.0, C, nColumnsB);
	return C;
}
