#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

void printarr(double arr[][5], int m, int n) {
	int i, j;
	for (i = 0; i<m; i++) {
		for (j = 0; j<n; j++) {
			printf("%lf ", arr[i][j]);
		}
		printf("\n");
	}
}
int main(int argc, char* argv[]) {
	int i;
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	double arr[4][5] = { { 1,2,3,4,5 },{ 2,3,1,4,8},{ 11,12,13,14,15 },{ 16,17,18,19,20 } };
	double softm[4][5];
	double soft[5], sm[4];
	
	if (rank == 0) {
		printf("ARRAY : \n");
		printarr(arr, 4, 5);
		fflush(stdout);
	}
	MPI_Scatter(arr, 5, MPI_DOUBLE, soft, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	double sum = 0;
	//printf("RANK : %d :", rank);
	for (i = 0; i<5; i++) {
		fflush(stdout);
		soft[i] = exp(soft[i]);
		//printf("rank : %d %lf ",rank,soft[i]);
		fflush(stdout);
		sum += soft[i];
	}
	printf("Rank : %d SUM : %lf ",rank, sum);
	MPI_Gather(&sum, 1, MPI_DOUBLE, sm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(sm, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (i = 0; i<5; i++) {
		soft[i] = soft[i] / sm[rank];
		//printf("%lf ", soft[i]);
	}
	MPI_Gather(soft, 5, MPI_DOUBLE, softm, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("ANSWER : \n");
		fflush(stdout);
		printarr(softm, 4, 5);
	}
	MPI_Finalize();
	return 0;
}