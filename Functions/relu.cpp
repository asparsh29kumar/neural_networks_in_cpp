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
	printf("SIZE : %d\n", size);
	double arr[4][5] = { { 1,-2,3,-4,5 },{ 6,-7,8,-9,10 },{ 11,-12,13,-14,15 },{ 16,-17,18,-19,20 } };
	double rel_mat[4][5];
	double rel;
	MPI_Scatter(arr, 1, MPI_DOUBLE, &rel, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("rank : %d %lf \n", rank, rel);
	if (rel<0) {
		rel = 0;
	}
	MPI_Gather(&rel, 1, MPI_DOUBLE, rel_mat, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printarr(rel_mat, 4, 5);
	}
	MPI_Finalize();
	return 0;
}