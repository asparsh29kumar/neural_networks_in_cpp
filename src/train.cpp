#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../trial.cpp"

using namespace std;

int* train(int** X,int* y,int** X_val,int** y_val,double learning_rate,double learning_rate_decay,double reg,int num_iters, int batch_size,int verbose,int num_train, int x_col,int y_row, int y_col)
{
    srand(time(NULL));
    double iterations_per_epoch;
    if(num_train/batch_size>1)
    {
        iterations_per_epoch = num_train/batch_size;
    }
    else
    {
        iterations_per_epoch = 1;
    }
    int *loss_history,*train_acc_history,*val_acc_history;
    int **X_batch,*y_batch,*sample_indices;
    int t1;
    for(int it = 0;it<num_iters;it++)
    {
        // sample_indices = np.random.choice(np.arange(num_train), batch_size)
        int *temp = (int*)calloc(sizeof(int),num_train);
        sample_indices = (int*)malloc(sizeof(int)*batch_size);
        for(int i = 0;i<batch_size;i++)
        {
            t1 = rand()%num_train;
            while(temp[t1] == 1)
            {
                t1 = rand()%num_train;
            }
            temp[t1] = 1;
            sample_indices[i] = t1;
        }
        // X_batch = X[sample_indices]
        X_batch = (int**)malloc(sizeof(int*)*batch_size);
        for(int i = 0;i<batch_size;i++)
        {
            X_batch[i] = (int*)malloc(sizeof(int)*x_col);
            for(int j = 0;j<x_col;j++)
            {
                X_batch[i][j] = X[sample_indices[i]][j];
            }
        }
        // y_batch = y[sample_indices]
        y_batch = (int*)malloc(sizeof(int)*batch_size);
        for(int i = 0;i<batch_size;i++)
        {
            y_batch[i] = y[sample_indices[i]];
        }
        // loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
        // double **grads_W1;
        // double **grads_W2;
        // double *grads_b1;
        // double *grads_b2;
        double loss_val = loss(reg,grads_W1,grads_W2,grads_b1,grads_b2,X,output_size,y,hidden_size);
        // loss_history.append(loss)
        int siz = sizeof(loss_history);
        loss_history = (double *)realloc(loss_history, siz+sizeof(double));
        loss_history[siz/sizeof(double)] = loss_val;
        for(int i = 0;i<input_size;i++)
        {
        	for(int j = 0 ; j < hidden_size ; j++)
        	{
        		W1[i][j] += (-learning_rate * grads_W1[i][j])
        	}
        }
        for(int i = 0;i<hidden_size;i++)
        {
        	for(int j = 0 ; j < output_size ; j++)
        	{
        		W2[i][j] += (-learning_rate * grads_W2[i][j])
        	}
        }
        for(int i = 0;i<hidden_size;i++)
        {
        	b1[i] += (-learning_rate * grads_b1[i]);
        }
        for(int i = 0;i<output_size;i++)
        {
        	b2[i] += (-learning_rate * grads_b2[i]);
        }
        // if verbose and it % 100 == 0:
        // print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        if(verbose == 1 & it%10 == 0):
        	printf("iteration %d / %d: loss %f\n",it,num_iters,loss_val);
        int *pred = (int*)malloc(sizeof(int)*num_inputs);
        if (it % iterations_per_epoch == 0)
        {
        	predict(X_batch,batch_size,x_col,pred);
	        int count = 0;
	        for(int i = 0;i<batch_size;i++)
	        {
	        	if(pred[i] == y_batch[i])
	        	{
	        		count++;
	        	}
	        }
	        count = 0;
	        double train_acc = count/batch_size;
        	predict(X_val,x_val_r,input_size,pred);
	        for(int i = 0;i<;i++)
	        {
	        	if(pred[i] == y_val[i])
	        	{
	        		count++;
	        	}
	        }
	        double val_acc = count/x_val_r;

	        int siz;
	        siz = sizeof(train_acc_history);
	        train_acc_history = (double*)realloc(train_acc_history,siz+sizeof(double));

	        siz = sizeof(val_acc_history);
	        val_acc_history = (double*)realloc(val_acc_history,siz+sizeof(double));

	        learning_rate += learning_rate_decay;
        }

        // train_acc = (self.predict(X_batch) == y_batch).mean()
        // val_acc = (self.predict(X_val) == y_val).mean()
        // train_acc_history.append(train_acc)
        // val_acc_history.append(val_acc)
        // learning_rate *= learning_rate_decay

    }
}
