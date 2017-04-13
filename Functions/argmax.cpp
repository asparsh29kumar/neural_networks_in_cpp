#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int m, n, i, j, k;
	int mat1[100][100];
	int mat2[100][100];
	int ans[100];
	if (rank == 0)
	{
		mat1[0][0] = 3; mat1[0][1] = 4; mat1[0][2] = 1;
		mat1[1][0] = 4; mat1[1][1] = 2; mat1[1][2] = 3;
		mat1[2][0] = 1; mat1[2][1] = 4; mat1[2][2] = 5;
		fflush(stdout);
		m = 3; n = 3;
		printf("Enter the matrix 1 :\n");
		for (i = 0; i<m; i++)
		{
			for (j = 0; j < n; j++)
			{
				printf("%d ", mat1[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int rb1[100];
	MPI_Scatter(mat1, 100, MPI_INT, rb1, 100, MPI_INT, 0, MPI_COMM_WORLD);
	int max_i;
	if (rank < m)
	{
		max_i = 0;
		for (i = 0; i < n; i++)
		{
			if (rb1[i] > rb1[max_i])
			{
				max_i = i;
			}
		}
	}
	MPI_Gather(&max_i, 1, MPI_INT, ans, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("ANSWER : \n");
		fflush(stdout);
		for (i = 0; i<m; i++)
		{
			printf("%d ",ans[i]);
		}
	}
	MPI_Finalize();
}