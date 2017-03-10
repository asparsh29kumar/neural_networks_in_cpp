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
void copy_matrix(double **arr1,int m,int n,double ***arr3){
	int i,j;
	double **arr2=new double*[m];
	for(i=0;i<m;i++){
		arr2[i]=new double[n];
	}
	for(i=0;i<m;i++){
		for(j=0;j<n;j++){
			arr2[i][j]=arr1[i][j];
		}
	}
	arr3=&arr2;
}

void mat_row_wise_sum(double **arr1,int m,int n,double **sm){
	double *sum=new double[m];
	int i,j;
	for(int i=0;i<m;i++){
		sum[i]=0;
		for(j=0;j<n;j++){
			sum[i]+=arr1[i][j];
		}
	}
	sm=&sum;
}

void transpose(double **arr1,int m,int n,double ***trans){
	int i,j;
	double **arr2=new double*[n];
	for(i=0;i<m;i++){
		arr2[i]=new double[m];
	}
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			arr2[i][j]=arr1[j][i];
		}
	}
	trans=&arr2;
}

void softmax(double** arr,int m1,int n1,double ***sftm){
	int sum,i,j;
	double **soft=new double*[m1];
	for(i=0;i<m1;i++){
		soft[i]=new double[n1];
	}

	for(i=0;i<m1;i++){
		sum=0;
		for(j=0;j<n1;j++){
			sum+=exp(arr[i][j]);
		}
		for(j=0;j<n1;j++){
			soft[i][j]=exp(arr[i][j])/sum;
		}
	}
	sftm=&soft;
}

void mat_element_wise_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2,double*** elematmul){
	if(m1!=m2||n1!=n2){
		cout<<"matrix dimension mismatch for element wise matrix multiplication"<<endl;
		return ;
	}
	int i,j;
	double **mul=new double*[m1];
	for(i=0;i<m1;i++){
		mul[i]=new double[n1];
	}
	for(i = 0;i<m1;i++){
        for(j = 0;j<n1;j++){
        	mul[i][j]=a1[i][j]*a2[i][j];
    	}    
	}
	elematmul=&mul;
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


void mat_addition(double** a1,double* a2,int m1,int n1,int m2,double*** matadd){
	if(m1!=m2){
		cout<<"matrix dimension mismatch for matrix addition"<<endl;
		return;
	}
	int i,j;
	double **add=new double*[m1];
	for(i=0;i<m1;i++){
		add[i]=new double[n1];
	}
	for(i=0;i<m1;i++){
		for(j=0;j<n1;j++){
			add[i][j]=a1[i][j]+a2[j];
		}
	}
	matadd=&add;
}

void mat_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2,double*** matmul){
	if(n1!=m2){
		cout<<"matrix dimension mismatch for matrix multiplication"<<endl;
		return;
	}
	int i,j,k;
	double **mul=new double*[m1];
	for(i=0;i<m1;i++){
		mul[i]=new double[n2];
	}
	for(i = 0;i<m1;i++){
        for(j = 0;j<n2;j++){
            for(k = 0; k < n1;k++){
                mul[i][j] += a1[i][k] * a2[k][j];
            }
    	}    
	}
	print_mat(mul,m1,n2);
	matmul=&mul;
}

void relu(double** a1,int m1,int n1,double*** relfn){
	int i,j,k;
	double **act=new double*[m1];
	for(i=0;i<m1;i++){
		act[i]=new double[n1];
	}
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
	relfn=&act;
}

int input_size = 4;
int hidden_size = 10;
int output_size = 3;//num_classes
int num_inputs = 5;

//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(10);
std::normal_distribution<double> distribution (0.0,1.0);

class TwoLayerNet{
	public:
	double **W1,*b1,**W2,*b2;
	TwoLayerNet(int input_size,int hidden_size,int output_size,float std=1e-4)
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
		for(int i=0;i<input_size;i++){
			for(int j=0;j<hidden_size;j++){
				W1[i][j]=distribution(generator);
			}
		}
		for(int i=0;i<hidden_size;i++){
			b1[i]=0;
		}
		for(int i=0;i<hidden_size;i++){
			for(int j=0;j<output_size;j++){
				W2[i][j]=distribution(generator);
			}
		}
		for(int i=0;i<output_size;i++){
			b2[i]=0;
		}
		printf("W1=\n");
		for(int i=0;i<input_size;i++){
			for(int j=0;j<hidden_size;j++){
				printf("%lf ",W1[i][j]);
			}
			printf("\n");
		}
		printf("W2=\n");
		for(int i=0;i<hidden_size;i++){
			for(int j=0;j<output_size;j++){
				printf("%lf ",W2[i][j]);
			}
			printf("\n");
		}
	}
	//TwoLayerNet init_toy_model()
	//{
	//	return TwoLayerNet(input_size, hidden_size, num_classes,std);
	//}
	void init_toy_data(double ***X,int **y)
	{
		int i,j;
		//y={0,1,2,2,1};
		double **arr=*X;
		int *temp=*y;
		for(i=0;i<num_inputs;i++){
			for(j=0;j<input_size;j++){
				arr[i][j]=i+j;
			}
		}
		temp[0]=0;temp[1]=1;temp[2]=2;temp[3]=2;temp[4]=1;
		X=&arr;
		y=&temp;
	}
	double loss(double reg,double ***grads_W1,double ***grads_W2,double **grads_b1,double **grads_b2,double **X,int output_size,int *y,int hidden_size){
		printf("At start of loss function\n");
		int i,j;
	    //z1 = X.dot(W1) + b1
	    double **mul=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			mul[i]=(double*)malloc(hidden_size*sizeof(double));
		}
		mat_multiply(X,W1,num_inputs,input_size,input_size,hidden_size,&mul);
		printf("mat_mul 1 done:::\n");
		print_mat(X,num_inputs,input_size);
		print_mat(W1,input_size,hidden_size);
		print_mat(mul,num_inputs,hidden_size);

		double **z1=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			z1[i]=(double*)malloc(hidden_size*sizeof(double));
		}

	    mat_addition(mul,b2,num_inputs,hidden_size,hidden_size,&z1);
	    printf("mat_add 2 done:::\n");
		print_mat(z1,num_inputs,hidden_size);

	    //a1 = np.maximum(0, z1) # pass through ReLU activation function
	    double **a1=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			a1[i]=(double*)malloc(hidden_size*sizeof(double));
		}

	    relu(z1,num_inputs,hidden_size,&a1);
	    printf("relu 3 done:::\n");
		print_mat(a1,num_inputs,hidden_size);
		
	    // scores = a1.dot(W2) + b2
	    double **mul2=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			mul2[i]=(double*)malloc(output_size*sizeof(double));
		}
		mat_multiply(X,W1,num_inputs,hidden_size,hidden_size,output_size,&mul2);

		printf("mat_mul 4 done:::\n");
		print_mat(mul2,num_inputs,output_size);

		double **score=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			score[i]=(double*)malloc(output_size*sizeof(double));
		}

	    mat_addition(mul2,b2,num_inputs,output_size,output_size,&score);


		printf("mat_add 5 done:::\n");
		print_mat(score,num_inputs,output_size);

	    double **probs=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			probs[i]=(double*)malloc(output_size*sizeof(double));
		}

	    softmax(score,num_inputs,output_size,&probs);


		printf("softmax 6 done:::\n");
		print_mat(probs,num_inputs,output_size);

	    //corect_logprobs = -np.log(probs[range(N), y])
	    
	    double *corect_logprobs=new double[num_inputs];
	    
	    for(i=0;i<num_inputs;i++){
	    	corect_logprobs[i]=-log(probs[i][y[i]]);
	    }//now shape of corect_logprobs=number of inputs;
	   
	    //data_loss = np.sum(corect_logprobs) / N
	    double data_loss=sum(corect_logprobs,num_inputs)/num_inputs;
	  	printf("data_loss=%lf\n",data_loss);
	    
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
		mat_element_wise_multiply(W1,W1,input_size,hidden_size,input_size,hidden_size,&elemul1);
		mat_element_wise_multiply(W2,W2,hidden_size,output_size,hidden_size,output_size,&elemul2);


		printf("mat_ele_mul 7 done:::\n");
		print_mat(elemul1,num_inputs,hidden_size);
		printf("mat_ele_mul 8 done:::\n");
		print_mat(elemul2,hidden_size,output_size);

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
	    copy_matrix(probs,num_inputs,output_size,&dscores);
	    for(i=0;i<num_inputs;i++){
	    	dscores[i][y[i]]-=1;
	    }
	    for(i=0;i<num_inputs;i++){
	    	for(j=0;j<output_size;j++){
	    		dscores[i][j]/=num_inputs;
	    	}
	    }
	    
	    //grads['W2'] = np.dot(a1.T, dscores)
	    //grads['b2'] = np.sum(dscores, axis=0) i.e. row wise sum
	    double **transa1=(double**)malloc(hidden_size*sizeof(double*));
		for(int i=0;i<hidden_size;i++)
		{
			transa1[i]=(double*)malloc(num_inputs*sizeof(double));
		}
		double **mul3=(double**)malloc(hidden_size*sizeof(double*));
		for(int i=0;i<hidden_size;i++)
		{
			mul3[i]=(double*)malloc(output_size*sizeof(double));
		}
		transpose(a1,num_inputs,hidden_size,&transa1);
		mat_multiply(transa1,dscores,hidden_size,num_inputs,num_inputs,output_size,&mul3);
		grads_W2=&mul3;
	    
	    double *rowsum=(double*)malloc(num_inputs*sizeof(double*));
	    mat_row_wise_sum(dscores,num_inputs,output_size,&rowsum);
	    grads_b2=&rowsum;

	    //dhidden = np.dot(dscores, W2.T)
	    double **transW2=(double**)malloc(output_size*sizeof(double*));
		for(int i=0;i<output_size;i++)
		{
			transa1[i]=(double*)malloc(hidden_size*sizeof(double));
		}
		double **dhidden=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			dhidden[i]=(double*)malloc(hidden_size*sizeof(double));
		}

	    mat_multiply(dscores,transW2,num_inputs,output_size,output_size,hidden_size,&dhidden);
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
		double **mul4=(double**)malloc(num_inputs*sizeof(double*));
		for(int i=0;i<num_inputs;i++)
		{
			mul4[i]=(double*)malloc(hidden_size*sizeof(double));
		}
		transpose(X,num_inputs,input_size,&transX);
		mat_multiply(transX,dhidden,num_inputs,input_size,input_size,hidden_size,&mul4);
		grads_W1=&mul4;

		double *rowsum2=(double*)malloc(num_inputs*sizeof(double*));
		mat_row_wise_sum(dhidden,num_inputs,hidden_size,&rowsum2);
		grads_b1=&rowsum2;
	    
	    //add regularization gradient contribution
	    //grads['W2'] += reg * W2
	    //grads['W1'] += reg * W1
	    for(i=0;i<hidden_size;i++){
	    	for(j=0;j<output_size;j++){
	    		*grads_W2[i][j]+=W2[i][j]*reg;
	    	}
	    }
	    for(i=0;i<num_inputs;i++){
	    	for(j=0;j<hidden_size;j++){
	    		*grads_W1[i][j]+=W1[i][j]*reg;
	    	}
	    }
	    return loss_f;
	}

};

int main(){

	double **grads_W1,**grads_W2,*grads_b1,*grads_b2;
	TwoLayerNet net(input_size, hidden_size, output_size,1e-1);
	print_mat(net.W1,input_size,hidden_size);print_mat(net.W2,hidden_size,output_size);

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
		for(int i=0;i<num_inputs;i++)
		{
			grads_W2[i]=(double*)malloc(output_size*sizeof(double));
		}
	grads_b1=(double*)malloc(input_size*sizeof(double*));
	grads_b2=(double*)malloc(input_size*sizeof(double*));

	net.init_toy_data(&X,&y);
	printf("X=\n");
	for(int i=0;i<num_inputs;i++){
		for(int j=0;j<input_size;j++){
			printf("%lf ",X[i][j]);
		}
		printf("\n");
	}
	printf("y=\n");
	for(int i=0;i<num_inputs;i++){
		printf("%d ",y[i]);
	}
	
	//print_mat(X,num_inputs,input_size);//print_mat(net.W1,input_size,hidden_size);print_mat(net.W2,hidden_size,output_size);
	double loss=net.loss(0.1,&grads_W1,&grads_W2,&grads_b1,&grads_b2,X,output_size,y,hidden_size);
	//double correct_loss = 1.30378789133;
	//cout<<"difference between your loss and correct loss is::"<<loss-correct_loss;
	
	return 0;
}
