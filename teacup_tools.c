/*
 * teacup_tools.c
 *
 *  Created on: Jul 22, 2012
 *      Author: Pasieka Manuel , mapa17@posgrado.upv.es
 */
#include <stdio.h>
#include <stdlib.h>

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
