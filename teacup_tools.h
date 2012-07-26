/*
 * teacup_tools.h
 *
 *  Created on: Jul 22, 2012
 *      Author: Pasieka Manuel , mapa17@posgrado.upv.es
 */

#ifndef TEACUP_TOOLS_H_
#define TEACUP_TOOLS_H_

extern struct timeval start;

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

void tick(void);
double tack(void);

#endif /* TEACUP_TOOLS_H_ */
