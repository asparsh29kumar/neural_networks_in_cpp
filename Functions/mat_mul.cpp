#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void main(int argc,char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int m, n, x, y, i, j, k;
	int mat1[100][100];
	int mat2[100][100];
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
		scanf("%d", &y);
		fflush(stdout);
		printf("Enter the matrix 2 :\n");
		for (i = 0; i<x; i++)
		{
			for (j = 0; j < y; j++)
			{
				mat2[i][j] = i + 1;
				printf("%d ", mat2[i][j]);
			}
			printf("\n");
		}
		fflush(stdout);
		if (n != x)
		{
			printf("This multiplication is not possible\n");
			exit(0);
		}
	}
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int rb1[100];
	MPI_Scatter(mat1, 100, MPI_INT, rb1, 100, MPI_INT, 0, MPI_COMM_WORLD);
	int rb2[100][100];
	MPI_Bcast(mat2, 100 * 100, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(mat1, 100 * 100, MPI_INT, 0, MPI_COMM_WORLD);
	int mul[100];
	if (rank < m)
	{
		for (i = 0; i < y; i++)
		{
			mul[i] = 0;
			for (j = 0; j < n; j++)
			{
				mul[i] += (rb1[j] * mat2[j][i]);
			}
		}
	}
	// int ans[100][100];
	MPI_Gather(mul, 100, MPI_INT, ans, 100, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		printf("ANSWER : \n");
		fflush(stdout);
		for (i = 0; i<m; i++)
		{
			for (j = 0; j < y; j++)
			{
				printf("%d ", ans[i][j]);
			}
			printf("\n");
		}
	}
	/*
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<y; j++)
		{
			ans[i][j] = 0;
			for (k = 0; k<n; k++)
				ans[i][j] += (mat1[i][k] * mat2[k][j]);
		}
	}
	printf("Solution :\n");
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<y; j++)
			printf("%d ", ans[i][j]);
		printf("\n");
	}
	*/
	MPI_Finalize();
}