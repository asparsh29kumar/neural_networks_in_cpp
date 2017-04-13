#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>

using namespace std;

void print_mat(double** arr,int m,int n){
	int i,j;
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			cout<<arr[i][j]<<" ";
		}
		cout<<endl;
	}
}
double sum(double arr[],int n){
	double sum=0;
	int i;
	for(i=0;i<n;i++){
		sum+=arr[i];
	}
	return sum;
}
void copy_matrix(double **arr1,int m,int n,double **arr2){
	int i,j;
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			arr2[i][j]=arr1[i][j];
		}
	}
}

void mat_row_wise_sum(double **arr1,int m,int n,double *sum){
	int i,j;
	for(int i=0;i<m;i++){
		sum[i]=0;
		for(j=0;j<n;j++){
			sum[i]+=arr1[i][j];
		}
	}
}

void transpose(double **arr1,int m,int n,double **trans){
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			trans[i][j]=arr1[j][i];
		}
	}
}

void softmax(double** arr,int m1,int n1,double **soft){
	int sum,i,j;

	for(i=0;i<m1;i++){
		sum=0;
		for(j=0;j<n1;j++){
			sum+=exp(arr[i][j]);
		}
		for(j=0;j<n1;j++){
			soft[i][j]=exp(arr[i][j])/sum;
		}
	}
}

void mat_element_wise_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2,double** mul){
	if(m1!=m2||n1!=n2){
		cout<<"matrix dimension mismatch for element wise matrix multiplication"<<endl;
		return ;
	}
	int i,j;
	for(i = 0;i<m1;i++){
        for(j = 0;j<n1;j++){
        	mul[i][j]=a1[i][j]*a2[i][j];
    	}
	}
}
double array_sum(double *arr,int n){
	int i;
	double sum=0;
	for(i=0;i<n;i++){
		sum+=arr[i];
	}
	return sum;
}
double mat_sum(double **arr,int m,int n){
	int i,j;
	double sum=0;
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			sum+=arr[i][j];
		}
	}
	return sum;
}


void mat_addition(double** a1,double* a2,int m1,int n1,int m2,double** matadd){
	if(n1!=m2){
		cout<<"matrix dimension mismatch for matrix addition"<<endl;
		return;
	}
	int i,j;
	for(i=0;i<m1;i++){
		for(j=0;j<n1;j++){
			matadd[i][j]=a1[i][j]+a2[j];
		}
	}
}

void mat_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2,double** matmul){
	if(n1!=m2){
		cout<<"matrix dimension mismatch for matrix multiplication"<<endl;
		return;
	}
	int i,j,k;
	for(i = 0;i<m1;i++){
        for(j = 0;j<n2;j++){
        	matmul[i][j] = 0;
            for(k = 0; k < n1;k++){
                matmul[i][j] += a1[i][k] * a2[k][j];
            }
    	}
	}
}

void relu(double** a1,int m1,int n1,double** act){
	int i,j,k;
	for(i = 0;i<m1;i++){
        for(j = 0;j<n1;j++){
            if(a1[i][j]<0){
            	act[i][j]=0;
            }
            else{
            	act[i][j]=a1[i][j];
            }
    	}
	}
}

void argmax(double**arr,int m1,int n1,int* maxindex){
	int i,j;
	int index;
	double max;
	for(i=0;i<m1;i++){
		index=0;
		max=arr[i][0];
		for (j=0;j<n1;j++){
			if(arr[i][j]>=max){
				max=arr[i][j];
				index=j;
			}
		}
		maxindex[i]=index;
	}
}

int input_size = 2;
int hidden_size = 10;
int output_size = 2;
int num_inputs = 4;

std::default_random_engine generator(10);
std::normal_distribution<double> distribution (0.0,1.0);

class TwoLayerNet{
	public:
	void predict(double** X,int num_inputs,int input_size,int* y_pred);
	void init_toy_data(double ***X,int **y);
	void train(double** X,int* y,double** X_val,int* y_val,double learning_rate,double learning_rate_decay,double reg,int num_iters, int batch_size,int verbose,int num_train, int x_col, double **grads_W1,double** grads_W2,double *grads_b1,double* grads_b2);
	double loss(double reg,double **grads_W1,double **grads_W2,double *grads_b1,double *grads_b2,double **X,int output_size,int *y,int hidden_size);
	double **W1,*b1,**W2,*b2;
	TwoLayerNet(int input_size,int hidden_size,int output_size,float std=1e-4);
};
TwoLayerNet::TwoLayerNet(int input_size,int hidden_size,int output_size,float std)
{
	W1=new double*[input_size];
	b1=new double[hidden_size];
	W2=new double*[hidden_size];
	b2=new double[output_size];
	for(int i=0;i<input_size;i++){
		W1[i]=new double[hidden_size];
	}
	for(int i=0;i<hidden_size;i++){
		W2[i]=new double[output_size];
	}
		for(int i=0;i<hidden_size;i++){
			b1[i]=0;
		}

		for(int i=0;i<output_size;i++){
			b2[i]=0;
		}
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
}
void TwoLayerNet::init_toy_data(double ***X,int **y)
{
	int i,j;
	//y={0,1,2,2,1};
	double **arr=*X;
	int *temp=*y;
	arr[0][0]=0;arr[0][1]=0;arr[1][0]=0;arr[1][1]=1;arr[2][0]=1;arr[2][1]=0;arr[3][0]=1;arr[3][1]=1;
	temp[0]=0;temp[1]=1;temp[2]=1;temp[3]=1;
	X=&arr;
	y=&temp;
}
double TwoLayerNet::loss(double reg,double **grads_W1,double **grads_W2,double *grads_b1,double *grads_b2,double **X,int output_size,int *y,int hidden_size){

	int i,j;
    //z1 = X.dot(W1) + b1
    double **mul=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		mul[i]=(double*)malloc(hidden_size*sizeof(double));
	}
	mat_multiply(X,W1,num_inputs,input_size,input_size,hidden_size,mul);

	double **z1=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		z1[i]=(double*)malloc(hidden_size*sizeof(double));
	}

    mat_addition(mul,b1,num_inputs,hidden_size,hidden_size,z1);

    //a1 = np.maximum(0, z1) # pass through ReLU activation function
    double **a1=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		a1[i]=(double*)malloc(hidden_size*sizeof(double));
	}

    relu(z1,num_inputs,hidden_size,a1);

    // scores = a1.dot(W2) + b2
    double **mul2=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		mul2[i]=(double*)malloc(output_size*sizeof(double));
	}
	mat_multiply(a1,W2,num_inputs,hidden_size,hidden_size,output_size,mul2);

	double **score=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		score[i]=(double*)malloc(output_size*sizeof(double));
	}

    mat_addition(mul2,b2,num_inputs,output_size,output_size,score);

    double **probs=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		probs[i]=(double*)malloc(output_size*sizeof(double));
	}

    softmax(score,num_inputs,output_size,probs);

    //corect_logprobs = -np.log(probs[range(N), y])

    double *corect_logprobs=new double[num_inputs];

    for(i=0;i<num_inputs;i++){
    	corect_logprobs[i]=-log(probs[i][y[i]]);
    }

    //data_loss = np.sum(corect_logprobs) / N
    double data_loss=sum(corect_logprobs,num_inputs)/num_inputs;

    //reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
	double **elemul1=(double**)malloc(input_size*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		elemul1[i]=(double*)malloc(hidden_size*sizeof(double));
	}
	double **elemul2=(double**)malloc(hidden_size*sizeof(double*));
	for(int i=0;i<hidden_size;i++)
	{
		elemul2[i]=(double*)malloc(output_size*sizeof(double));
	}
	mat_element_wise_multiply(W1,W1,input_size,hidden_size,input_size,hidden_size,elemul1);
	mat_element_wise_multiply(W2,W2,hidden_size,output_size,hidden_size,output_size,elemul2);

    double reg_loss=0.5*reg*mat_sum(elemul1,input_size,hidden_size)+0.5*reg*mat_sum(elemul2,hidden_size,output_size);

    double loss_f = data_loss + reg_loss;

    cout<<"data_loss="<<data_loss<<" reg_loss="<<reg_loss<<" loss_f="<<loss_f<<endl;

    //Backward pass: compute gradients
    //dscores = probs
    //dscores[range(N),y] -= 1
    //dscores /= N
    double **dscores=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		dscores[i]=(double*)malloc(output_size*sizeof(double));
	}
    copy_matrix(probs,num_inputs,output_size,dscores);
    for(i=0;i<num_inputs;i++){
    	dscores[i][y[i]]-=1;
    }
    for(i=0;i<num_inputs;i++){
    	for(j=0;j<output_size;j++){
    		dscores[i][j]/=num_inputs;
    	}
    }
    //grads['W2'] = np.dot(a1.T, dscores)
    //grads['b2'] = np.sum(dscores, axis=0)
    double **transa1=(double**)malloc(hidden_size*sizeof(double*));
	for(int i=0;i<hidden_size;i++)
	{
		transa1[i]=(double*)malloc(num_inputs*sizeof(double));
	}
	transpose(a1,num_inputs,hidden_size,transa1);
	mat_multiply(transa1,dscores,hidden_size,num_inputs,num_inputs,output_size,grads_W2);
    mat_row_wise_sum(dscores,num_inputs,output_size,grads_b2);

    //dhidden = np.dot(dscores, W2.T)
    double **transW2=(double**)malloc(output_size*sizeof(double*));
	for(int i=0;i<output_size;i++)
	{
		transW2[i]=(double*)malloc(hidden_size*sizeof(double));
	}
	double **dhidden=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		dhidden[i]=(double*)malloc(hidden_size*sizeof(double));
	}
	transpose(W2,hidden_size,output_size,transW2);
    mat_multiply(dscores,transW2,num_inputs,output_size,output_size,hidden_size,dhidden);
    //backprop the ReLU non-linearity
    for(i=0;i<num_inputs;i++){
    	for(j=0;j<hidden_size;j++){
    		if(a1[i][j]<=0){
    			dhidden[i][j]=0;
    		}
    	}
    }

    //finally into W,b
    //grads['W1'] = np.dot(X.T, dhidden)
    //grads['b1'] = np.sum(dhidden, axis=0)
    double **transX=(double**)malloc(input_size*sizeof(double*));
	for(int i=0;i<input_size;i++)
	{
		transX[i]=(double*)malloc(num_inputs*sizeof(double));
	}

	transpose(X,num_inputs,input_size,transX);
	mat_multiply(transX,dhidden,input_size,num_inputs,num_inputs,hidden_size,grads_W1);
	mat_row_wise_sum(dhidden,num_inputs,hidden_size,grads_b1);
    //add regularization gradient contribution
    //grads['W2'] += reg * W2
    //grads['W1'] += reg * W1
    for(i=0;i<hidden_size;i++){
    	for(j=0;j<output_size;j++){
    		grads_W2[i][j]+=W2[i][j]*reg;
    	}
    }

    for(i=0;i<input_size;i++){
    	for(j=0;j<hidden_size;j++){
    		grads_W1[i][j]+=W1[i][j]*reg;
    	}
    }
    return loss_f;
}

void TwoLayerNet::predict(double** X,int num_inputs,int input_size,int* y_pred){
	int i,j;
    //z1 = X.dot(W1) + b1
    double **mul=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		mul[i]=(double*)malloc(hidden_size*sizeof(double));
	}
	mat_multiply(X,W1,num_inputs,input_size,input_size,hidden_size,mul);

	double **z1=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		z1[i]=(double*)malloc(hidden_size*sizeof(double));
	}

    mat_addition(mul,b1,num_inputs,hidden_size,hidden_size,z1);
    //a1 = np.maximum(0, z1) # pass through ReLU activation function
    double **a1=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		a1[i]=(double*)malloc(hidden_size*sizeof(double));
	}

    relu(z1,num_inputs,hidden_size,a1);

    // scores = a1.dot(W2) + b2
    double **mul2=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		mul2[i]=(double*)malloc(output_size*sizeof(double));
	}
	mat_multiply(a1,W2,num_inputs,hidden_size,hidden_size,output_size,mul2);

	double **score=(double**)malloc(num_inputs*sizeof(double*));
	for(int i=0;i<num_inputs;i++)
	{
		score[i]=(double*)malloc(output_size*sizeof(double));
	}

    mat_addition(mul2,b2,num_inputs,output_size,output_size,score);

	argmax(score,num_inputs,output_size,y_pred);
	print_mat(score,num_inputs,output_size);
	for(i = 0;i < num_inputs;i++)
    	printf("%d ",y_pred[i]);
	printf("\n");

}
void TwoLayerNet::train(double** X,int* y,double** X_val,int* y_val,double learning_rate,double learning_rate_decay,double reg,int num_iters, int batch_size,int verbose,int num_train, int x_col, double **grads_W1,double** grads_W2,double *grads_b1,double* grads_b2){
    srand(time(NULL));
    int iterations_per_epoch;
    if(num_train/batch_size>1)
    {
        iterations_per_epoch = num_train/batch_size;
    }
    else
    {
        iterations_per_epoch = 1;
    }
    printf("num_train : %d batch_size : %d iterations_per_epoch : %d\n",num_train,batch_size,iterations_per_epoch);
    double *loss_history,*train_acc_history,*val_acc_history;
    double **X_batch;int* sample_indices,*y_batch;
    int t1;
    for(int it = 0;it<num_iters;it++)
    {

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
        X_batch = (double**)malloc(sizeof(double*)*batch_size);
        for(int i = 0;i<batch_size;i++)
        {
            X_batch[i] = (double*)malloc(sizeof(double)*x_col);
            for(int j = 0;j<x_col;j++)
            {
                X_batch[i][j] = X[sample_indices[i]][j];
            }
        }
        y_batch = (int*)malloc(sizeof(int)*batch_size);
        for(int i = 0;i<batch_size;i++)
        {
            y_batch[i] = y[sample_indices[i]];
        }
        double loss_val = loss(reg,grads_W1,grads_W2,grads_b1,grads_b2,X,output_size,y,hidden_size);
        for(int i = 0;i<input_size;i++)
        {
        	for(int j = 0 ; j < hidden_size ; j++)
        	{
        		W1[i][j] += (-learning_rate * grads_W1[i][j]);
        	}
        }
        for(int i = 0;i<hidden_size;i++)
        {
        	for(int j = 0 ; j < output_size ; j++)
        	{
        		W2[i][j] += (-learning_rate * grads_W2[i][j]);
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
        if(verbose == 1){
        	printf("\nIteration %d / %d: loss %f\n\n",it,num_iters,loss_val);
        }
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

        	predict(X_val,num_inputs,input_size,pred);
	        for(int i = 0;i<num_inputs;i++)
	        {
	        	if(pred[i] == y_val[i])
	        	{
	        		count++;
	        	}
	        }

	        double val_acc = (double)count/(double)num_inputs;
	        printf("VALIDATION ACCURACY :: %f \n",val_acc);

	        learning_rate += learning_rate_decay;
        }
	}
}
int main(){

	double **grads_W1,**grads_W2,*grads_b1,*grads_b2;
	TwoLayerNet net(input_size, hidden_size, output_size,1e-1);

	double **X=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			X[i]=(double*)malloc(input_size*sizeof(double));
		}
	int *y=(int*)malloc(num_inputs*sizeof(int));
	grads_W1=(double**)malloc(hidden_size*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			grads_W1[i]=(double*)malloc(hidden_size*sizeof(double));
		}
	grads_W2=(double**)malloc(hidden_size*sizeof(double*));
		for(int i=0;i<hidden_size;i++)
		{
			grads_W2[i]=(double*)malloc(output_size*sizeof(double));
		}
	grads_b1=(double*)malloc(input_size*sizeof(double*));
	grads_b2=(double*)malloc(input_size*sizeof(double*));

	net.init_toy_data(&X,&y);
	//loss(reg,grads_W1,grads_W2,grads_b1,grads_b2,X,output_size,y,hidden_size);
	double loss=net.loss(1e-5,grads_W1,grads_W2,grads_b1,grads_b2,X,output_size,y,hidden_size);
	double correct_loss = 1.30378789133;
	printf("\n\n\n\nstarting TRAIN!!!\n\n\n");
	//net.train(X,y,X,y,0.01,0.001,1e-5,10,1,1,4,input_size,grads_W1,grads_W2,grads_b1,grads_b2);
	net.train(X,y,X,y,0.00001,0.000001,1e-5,100,1,1,4,input_size,grads_W1,grads_W2,grads_b1,grads_b2);
//(learning_rate,learning_rate_decay,reg,num_iters,batch_size,verbose,num_train,x_col,grads_W1,grads_W2,grads_b1,grads_b2){
	int* pred=(int*)malloc(num_inputs*sizeof(int));

	net.predict(X,num_inputs,input_size,pred);

	printf("predictions::");
	for(int i=0;i<num_inputs;i++){
		printf("pred[%d]=%d\n",i,pred[i]);
	}
	printf("\n\n");
	return 0;
}