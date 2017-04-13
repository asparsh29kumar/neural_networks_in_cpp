#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int m, n, x, i, j, k;
	int mat1[100][100];
	int mat2[100];
	int ans[100][100];
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
		fflush(stdout);
		printf("Enter size of matrix 2 :");
		fflush(stdout);
		scanf("%d", &x);
		fflush(stdout);
		printf("Enter the matrix 2 :\n");
		for (i = 0; i<x; i++)
		{
			mat2[i] = i;
			printf("%d ", mat2[i]);
		}
		printf("\n");
		fflush(stdout);
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int rb1[100];
	MPI_Scatter(mat1, 100, MPI_INT, rb1, 100, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(mat2, 100, MPI_INT, 0, MPI_COMM_WORLD);
	int sum[100];
	if (rank < m)
	{
		for (i = 0; i < n; i++)
		{
			sum[i] = rb1[i] + mat2[i];
		}
	}
	MPI_Gather(sum, 100, MPI_INT, ans, 100, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("ANSWER : \n");
		fflush(stdout);
		for (i = 0; i<m; i++)
		{
			for (j = 0; j < n; j++)
			{
				printf("%d ", ans[i][j]);
			}
			printf("\n");
		}
	}
	MPI_Finalize();
}