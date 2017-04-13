#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include "mpi.h"
#define SIZ 20
using namespace std;
int input_size = 2;
int hidden_size = 20;
int output_size = 2;
int num_inputs = 4;
double W1[SIZ][SIZ], b1[SIZ] = {0}, W2[SIZ][SIZ], b2[SIZ];

std::default_random_engine generator(10);
std::normal_distribution<double> distribution(0.0, 1.0);


void relu(double arr[][SIZ], int m1, int n1, double rel_mat[][SIZ], int rank) {
    double rel[SIZ];
    MPI_Scatter(arr, SIZ, MPI_DOUBLE, &rel, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank<m1) {
        for(int i = 0;i<n1;i++)
            if(rel[i]<0)
                rel[i] = 0;
    }
    MPI_Gather(rel, SIZ, MPI_DOUBLE, rel_mat, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void softmax(double arr[][SIZ], int m1, int n1, double softm[][SIZ], int rank) {
    int i;
    double soft[SIZ], sm[SIZ];

    MPI_Scatter(arr, SIZ, MPI_DOUBLE, soft, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sum = 0;
    for (i = 0; i<n1; i++) {
        soft[i] = exp(soft[i]);
        sum += soft[i];
    }
    MPI_Gather(&sum, 1, MPI_DOUBLE, sm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(sm, m1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i<n1; i++) {
        soft[i] = soft[i] / sm[rank];
    }
    MPI_Gather(soft, SIZ, MPI_DOUBLE, softm, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void mat_element_wise_multiply(double mat1[][SIZ], double mat2[][SIZ], int m, int n, int x, int y, double ans[][SIZ], int rank) {
    double rb1[SIZ];
    double rb2[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat2, SIZ, MPI_DOUBLE, rb2, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mul[SIZ];
    if (rank < m)
    {
        for (int i = 0; i < y; i++)
        {
            mul[i] = rb1[i] * rb2[i];
        }
    }
    MPI_Gather(mul, SIZ, MPI_DOUBLE, ans, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void printarr(double mul[][SIZ],int m,int n,char *name)
{
    printf("%s\n",name);
    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j< n;j++)
            printf("%lf ",mul[i][j]);
        printf("\n");
    }
}

void mat_addition(double mat1[][SIZ], double mat2[SIZ], int m, int n, int x, double matadd[][SIZ], int rank) {
    // double ans[SIZ][SIZ];
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat2, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double sum[SIZ];
    if (rank < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum[i] = rb1[i] + mat2[i];
        }
    }
    MPI_Gather(sum, SIZ, MPI_DOUBLE, matadd, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void mat_multiply(double mat1[][SIZ], double mat2[][SIZ], int m, int n, int x, int y, double ans[][SIZ],int rank) {
    if (n != x)
    {
        printf("Error matrix size mismatch\n");
        return;
    }
    int i, j, k;
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double rb2[SIZ][SIZ];
    MPI_Bcast(mat2, SIZ * SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat1, SIZ * SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mul[SIZ];
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
    // int ans[SIZ][SIZ];
    MPI_Gather(mul, SIZ, MPI_DOUBLE, ans, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

double mat_sum(double mat1[][SIZ], int m, int n,int rank) {
    double ans;
    double rb[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank<m)
    {
        for(int i = 1;i<n;i++)
        {
            rb[0] += rb[i];
        }
    }
    MPI_Reduce(&rb[0], &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
        return ans;
    else
        return 0;
}

double sum(double arr[], int n) {
    double sum = 0;
    int i;
    for (i = 0; i<n; i++) {
        sum += arr[i];
    }
    return sum;
}

void transpose(double arr1[][SIZ], int m, int n, double trans[][SIZ]) {
    int i, j;
    for (i = 0; i<n; i++) {
        for (j = 0; j<m; j++) {
            trans[i][j] = arr1[j][i];
        }
    }
}


void mat_col_wise_add(double arr1[][SIZ], int m, int n, double *sum) {
    
    int i, j;
    for (int i = 0; i<n; i++) {
        sum[i] = 0;
        for (j = 0; j<m; j++) {
            sum[i] += arr1[j][i];
        }
    }
    
}

void copy_matrix(double arr[][SIZ], int m, int n, double copy_mat[][SIZ], int rank) {
    double cpy[SIZ][SIZ];
    MPI_Scatter(arr, SIZ, MPI_DOUBLE, cpy, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(cpy, SIZ, MPI_DOUBLE, copy_mat, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void argmax(double mat1[][SIZ], int m, int n, int* ans,int rank) {
    int i, j, k;
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
}

void init_toy_data(double X[][SIZ], int *y)
{
    int i, j;
    X[0][0]=0;X[0][1]=0;X[1][0]=0;X[1][1]=1;X[2][0]=1;X[2][1]=0;X[3][0]=1;X[3][1]=1;
    y[0] = 0; y[1] = 1; y[2] = 1; y[3] = 0;
}
double loss(double reg, double grads_W1[][SIZ], double grads_W2[][SIZ], double *grads_b1, double *grads_b2, double X[][SIZ], int output_size, int *y, int hidden_size,int rank) {
    if(rank == 0)printf("In loss \n");
    fflush(stdout);
    int i, j;
    fflush(stdout);
    double mul[SIZ][SIZ];
    mat_multiply(X, W1, num_inputs, input_size, input_size, hidden_size, mul, rank);
    double z1[SIZ][SIZ];
    mat_addition(mul, b1, num_inputs, hidden_size, hidden_size, z1,rank);
    double a1[SIZ][SIZ];
    relu(z1, num_inputs, hidden_size, a1, rank);
    double mul2[SIZ][SIZ];
    mat_multiply(a1, W2, num_inputs, hidden_size, hidden_size, output_size, mul2,rank);
    double score[SIZ][SIZ];
    mat_addition(mul2, b2, num_inputs, output_size, output_size, score,rank);
    double probs[SIZ][SIZ];
    softmax(score, num_inputs, output_size, probs, rank);
    double corect_logprobs[5];
    if (rank == 0)
    {
        for (i = 0; i < num_inputs; i++) {
            corect_logprobs[i] = -log(probs[i][y[i]]);
        }
    }
    double data_loss;
    double elemul1[SIZ][SIZ];
    double elemul2[SIZ][SIZ];
    mat_element_wise_multiply(W1, W1, input_size, hidden_size, input_size, hidden_size, elemul1,rank);
    mat_element_wise_multiply(W2, W2, hidden_size, output_size, hidden_size, output_size, elemul2,rank);
    double reg_loss;
    double loss_f;
    if (rank == 0)
    {
        data_loss = sum(corect_logprobs, num_inputs) / num_inputs;
    }
    reg_loss = 0.5*reg*mat_sum(elemul1, input_size, hidden_size, rank) + 0.5*reg*mat_sum(elemul2, hidden_size, output_size, rank);
    if(rank == 0)
    {
        loss_f = data_loss + reg_loss;
        cout << "data_loss=" << data_loss << " reg_loss=" << reg_loss << " loss_f=" << loss_f << endl;
    }
    double dscores[SIZ][SIZ];
    copy_matrix(probs, num_inputs, output_size, dscores,rank);
    if (rank == 0)
    {
        for (i = 0; i < num_inputs; i++) {
            dscores[i][y[i]] -= 1;
        }
        for (i = 0; i < num_inputs; i++) {
            for (j = 0; j < output_size; j++) {
                dscores[i][j] /= num_inputs;
            }
        }
    }
    double transa1[SIZ][SIZ];
    transpose(a1, num_inputs, hidden_size, transa1);
    mat_multiply(transa1, dscores, hidden_size, num_inputs, num_inputs, output_size, grads_W2,rank);
    mat_col_wise_add(dscores, num_inputs, output_size, grads_b2);
    double transW2[SIZ][SIZ];
    double dhidden[SIZ][SIZ];
    transpose(W2, hidden_size, output_size, transW2);
    mat_multiply(dscores, transW2, num_inputs, output_size, output_size, hidden_size, dhidden,rank);
    if (rank == 0)
    {
        for (i = 0; i < num_inputs; i++) {
            for (j = 0; j < hidden_size; j++) {
                if (a1[i][j] <= 0) {
                    dhidden[i][j] = 0;
                }
            }
        }
    }
    double transX[SIZ][SIZ];

    transpose(X, num_inputs, input_size, transX);
    mat_multiply(transX, dhidden, input_size, num_inputs, num_inputs, hidden_size, grads_W1,rank);
    mat_col_wise_add(dhidden, num_inputs, hidden_size, grads_b1);
    if (rank == 0)
    {
        for (i = 0; i < hidden_size; i++) {
            for (j = 0; j < output_size; j++) {
                grads_W2[i][j] += W2[i][j] * reg;
            }
        }

        for (i = 0; i < input_size; i++) {
            for (j = 0; j < hidden_size; j++) {
                grads_W1[i][j] += W1[i][j] * reg;
            }
        }
    }
    if(rank == 0)
    {
        return loss_f;
    }
    return 0;
}

void predict(double X[][SIZ], int num_inputs, int input_size, int* y_pred,int rank) {
    int i, j;
    double mul[SIZ][SIZ];
    mat_multiply(X, W1, num_inputs, input_size, input_size, hidden_size, mul,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double z1[SIZ][SIZ];
    mat_addition(mul, b1, num_inputs, hidden_size, hidden_size, z1,rank);   
    double a1[SIZ][SIZ];
    relu(z1, num_inputs, hidden_size, a1,rank);
    double mul2[SIZ][SIZ];
    mat_multiply(a1, W2, num_inputs, hidden_size, hidden_size, output_size, mul2,rank);
    double score[SIZ][SIZ];
    mat_addition(mul2, b2, num_inputs, output_size, output_size, score,rank);
    argmax(score, num_inputs, output_size, y_pred,rank);
    if(rank == 0)
    {
        printarr(score,num_inputs,output_size,"score");
        for(i = 0;i < num_inputs;i++)
            printf("%d ",y_pred[i]);
        printf("\n");
    }
}

void train(double X[][SIZ], int* y, double X_val[][SIZ], int* y_val, double learning_rate, double learning_rate_decay, double reg, int num_iters, int batch_size, int verbose, int num_train, int x_col, double grads_W1[][SIZ], double grads_W2[][SIZ], double *grads_b1, double* grads_b2,int rank) {
    if(rank == 0)
        printf("INSIDE TRAIN !!! Successfully!!!\n");
    srand(time(NULL));
    int iterations_per_epoch;
    if (rank == 0)
    {
        if (num_train / batch_size > 1)
        {
            iterations_per_epoch = num_train / batch_size;
        }
        else
        {
            iterations_per_epoch = 1;
        }
        printf("num_train : %d batch_size : %d iterations_per_epoch : %d\n",num_train,batch_size,iterations_per_epoch);
    }
    MPI_Bcast(&iterations_per_epoch,1,MPI_INT,0,MPI_COMM_WORLD);
    double X_batch[SIZ][SIZ];
    int sample_indices[SIZ], y_batch[SIZ];
    int t1;

    for (int it = 0; it<num_iters; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)printf("rank : %d it : %d iterations_per_epoch : %d\n", rank, it,iterations_per_epoch);
        int pred[5];
        int *temp = (int*)calloc(sizeof(int),num_train);
        if (rank == 0)
        {
            printf("SAMPLE INDICES :: \n");
            for (int i = 0; i < batch_size; i++)
            {
                t1 = rand() % num_train;
                while (temp[t1] == 1)
                {
                    t1 = rand() % num_train;
                }
                temp[t1] = 1;
                sample_indices[i] = t1;
            }
            for(int i = 0;i<batch_size;i++)
                printf("%d ",sample_indices[i]);
            printf("\nbatch_size : %d num_train : %d\n",batch_size,num_train);
            fflush(stdout);
            for (int i = 0; i < batch_size; i++)
            {
                for (int j = 0; j < x_col; j++)
                {
                    X_batch[i][j] = X[sample_indices[i]][j];
                }
            }
            printarr(X_batch,batch_size,x_col,"X_batch");
            for (int i = 0; i < batch_size; i++)
            {
                y_batch[i] = y[sample_indices[i]];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        double loss_val = loss(reg, grads_W1, grads_W2, grads_b1, grads_b2, X, output_size, y, hidden_size, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            for (int i = 0; i < input_size; i++)
            {
                for (int j = 0; j < hidden_size; j++)
                {
                    W1[i][j] += (-learning_rate * grads_W1[i][j]);
                }
            }
            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    W2[i][j] += (-learning_rate * grads_W2[i][j]);
                }
            }
            for (int i = 0; i < hidden_size; i++)
            {
                b1[i] += (-learning_rate * grads_b1[i]);
            }
            for (int i = 0; i < output_size; i++)
            {
                b2[i] += (-learning_rate * grads_b2[i]);
            }
            if (verbose == 1) {
                printf("\nIteration %d / %d: loss %f\n\n", it, num_iters, loss_val);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (it % iterations_per_epoch == 0)
        {
            predict(X_batch, batch_size, x_col, pred, rank);
            int count = 0;
            if (rank == 0)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    if (pred[i] == y_batch[i])
                    {
                        count++;
                    }
                }
                count = 0;
                double train_acc = count / batch_size;
            }

            predict(X_val, num_inputs, input_size, pred, rank);
            if (rank == 0)
            {
                for (int i = 0; i < num_inputs; i++)
                {
                    if (pred[i] == y_val[i])
                    {
                        count++;
                    }
                }
            }

            double val_acc = (double)count / (double)num_inputs;
            if (rank == 0)
            {
                printf("VALIDATION ACCURACY :: %f \n", val_acc);
            }
            learning_rate += learning_rate_decay;
        }
    }
}




int main(int argc,char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                double temp = fabs(distribution(generator));
                while (temp > 1)
                    temp = temp - 1;
                W1[i][j] = temp;
            }
        }
        for (int i = 0; i < hidden_size; i++) {
            b1[i] = 0;
        }
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                double temp = fabs(distribution(generator));
                while (temp > 1)
                    temp = temp - 1;
                W2[i][j] = temp;
            }
        }
        for (int i = 0; i < output_size; i++) {
            b2[i] = 0;
        }
    }
    int pred[5];

    double grads_W1[SIZ][SIZ], grads_W2[SIZ][SIZ], grads_b1[SIZ], grads_b2[SIZ];
    double X[SIZ][SIZ];
    int y[SIZ];
    if(rank == 0)
    {
        init_toy_data(X,y);
    }
    double los = loss(1e-5, grads_W1, grads_W2, grads_b1, grads_b2, X, output_size, y, hidden_size, rank);

    fflush(stdout);
    double correct_loss = 1.30378789133;
    if (rank == 0) {
        printf("\n\n\n\nstarting TRAIN!!!\n\n\n");
    }
    train(X, y, X, y, 0.01, 0.1, 1e-5, 20, 1, 1, 1, input_size, grads_W1, grads_W2, grads_b1, grads_b2, rank);
    predict(X, num_inputs, input_size, pred, rank);
    if (rank == 0)
    {
        printf("predictions::");
        for (int i = 0; i < num_inputs; i++) {
            printf("pred[%d]=%d\n", i, pred[i]);
        }
        printf("\n\n");
    }
    MPI_Finalize();
    return 0;
}