/*
 * teacup_tools.c
 *
 *  Created on: Jul 22, 2012
 *      Author: Pasieka Manuel , mapa17@posgrado.upv.es
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <string.h>

#include "teacup_tools.h"

struct timeval start;
char *b = NULL;

void tick(void){
	gettimeofday(&start, NULL);
}

double tack(void){
	struct timeval end;
	if( gettimeofday(&end, NULL) != 0){
		printf("Getting time failed!\n");
	}
//	printf("S %d:%d , E %d:%d\n", start.tv_sec, start.tv_usec, end.tv_sec, end.tv_usec);
	return (double) end.tv_sec-start.tv_sec + ( end.tv_usec - start.tv_usec ) / 1000000.0;
}

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

	//printf("Joining [%d] compressed Matrices with [%d] total Columns ...", nMatrices, nTotalColumns);

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

	//printf("Copy references ...\n");
	k = 0;
	for(i = 0; i < nMatrices; i++){
		for(j = 0; j < Cii[i]->columns; j++){
			C->nNoneZero[k+j] = Cii[i]->nNoneZero[j];
			C->NoneZero[k+j] = Cii[i]->NoneZero[j];
			C->nZero[k+j] = Cii[i]->nZero[j];
			C->Zero[k+j] = Cii[i]->Zero[j];
		}
		k+=j;

		assert( (k <= nTotalColumns) && "Too many columns!");
	}

	//printf("Finished joining matrices!\n");
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
			if(M[j + i*cM->columns] == 0.0){
				start = i;
				end = i+1;
				while( (end < cM->rows) && (M[j + end*cM->columns] == 0.0) ) end++;
				Z[j][cM->nZero[j]*2 + 0] = start;
				Z[j][cM->nZero[j]*2 + 1] = end;
				cM->nZero[j]++;
				i=end-1;
				//printf("zB[%d] [%d-%d]\n",j, start, end);
			} else {
				NZ[j][cM->nNoneZero[j]] = M[j + i*cM->columns];
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

	//printf("Uncompressing Matrix [%d x %d] ...",cM->rows, cM->columns);

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

	//printf("Finished!\n");
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

	//printf("Sending compressed matrix, columns [%d-%d] to node [%d] ...\n",startColumn, endColumn, destID);

	for(i = startColumn; i < endColumn; i++)
	{
		nTotalNoneZero += cM->nNoneZero[i];
		nTotalZero += cM->nZero[i];
	}

	//Be sure to have enough buffer
	bSize = sizeof(int)*(nTotalZero*2 + 2*(endColumn-startColumn)) + sizeof(double) * ( nTotalNoneZero );
	b = malloc( bSize );

	//printf("Reserving buffer of size [%d] for nTotalNZ[%d], nTotalZ[%d]\nPacking ...\n", bSize, nTotalNoneZero, nTotalZero);

	position=0;
	MPI_Pack(&(cM->nNoneZero[startColumn]), endColumn-startColumn, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->NoneZero[startColumn][0]), nTotalNoneZero, MPI_DOUBLE, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->nZero[startColumn]), endColumn-startColumn, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );
	MPI_Pack(&(cM->Zero[startColumn][0]), nTotalZero*2, MPI_INT, b, (int)bSize, &position, MPI_COMM_WORLD );

	//printf("Pack position [%d]\nSending ...",position);

	assert( (position == bSize) && "Something went wrong during packing!" );

	MPI_Send(&position, 1, MPI_INT, destID, 0, MPI_COMM_WORLD);
	MPI_Send(b, position, MPI_PACKED, destID, 0, MPI_COMM_WORLD);
	//printf("Finished sending compressed Matrix!\n");
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

	//printf("Receiving compressed matrix [%d x %d] from node [%d] ...\n",nColumns, nRows, srcID);

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

	//printf("Will need a receive buffer of size [%d]\n", bSize);

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


	//printf("Finished unpacking at position [%d]\n", position);

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

	//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nRowsA, nColumnsB, n, 1.0, A, n, B, nColumnsB, 1.0, C, nColumnsB);

	int i,j,k;

	#pragma omp parallel for schedule(runtime)
	for(i=0; i < nRowsA; i++){
		for(j=0; j<nColumnsB; j++){
			C[j + i*nColumnsB] = 0.0;
			for(k=0; k<n; k++){
				C[j + i*nColumnsB] += A[k + i*n] * B[j + k*nColumnsB];
			}
		}
	}

	return C;
}
