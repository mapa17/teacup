#include <stdio.h>
#include <stdlib.h>
#include <mkl_blas.h>
#include <ctimer.h>


int main(int argc, char *argv[]) {

  //int n, nb, N, lda;
  double *A, *B, *C, *D;
  int N; 
  int i, j, k;
  double elapsed,scpu, ucpu;

  // Input data
  N = atoi(argv[1]);
  // Allocating memory and filling data
      A = (double *) malloc( N * N * sizeof(double) );
      B = (double *) malloc( N * N * sizeof(double) );
      C = (double *) malloc( N * N * sizeof(double) );
      for(i=0; i < N; i++ ){
         for(j=0; j < N; j++){
  	    A[i+N*j] = 1;//((double) rand())/RAND_MAX;
  	    B[i+N*j] = 1;
  	    C[i+N*j] = 0;
         }
      }
  

  double alpha=1, beta=1;
  int m=N, n=N, lda=N, ldb=N, ldc=N;
  char transa='n', transb='n';
  k=N;
  ctimer_(&elapsed, &scpu, &ucpu);
  dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  ctimer_(&elapsed, &scpu, &ucpu);

  printf("%f segundos\n", elapsed);
/*
  printf("row \ta \tc\n");
  for (i=0; i<N; i++) {
     for (j=0; j<N; j++) {
		 printf("%7.3f ", C[i+N*j]);
     }
     printf("\n");
  }
*/
  free(A);
  free(B);
  free(C);
  return 0;
}
