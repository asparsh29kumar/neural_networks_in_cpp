#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int m, n, x, y, i, j, k;
	int mat1[10][10] = { 0 };
	int ans;
	if (rank == 0)
	{
		printf("Enter size of matrix 1 :");
		fflush(stdout);
		scanf("%d", &m);
		scanf("%d", &n);
		printf("Enter the matrix 1 :\n");
		for (i = 0; i<m; i++)
		{
			for (j = 0; j < n; j++)
			{
				mat1[i][j] = i + j;
				printf("%d ", mat1[i][j]);
			}
			printf("\n");
		}
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int rb;
	MPI_Scatter(mat1, 1, MPI_INT, &rb, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// printf("Rank : %d : %d", rank, rb);
	MPI_Reduce(&rb, &ans, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("ANSWER : %d\n", ans);
	}
	MPI_Finalize();
}