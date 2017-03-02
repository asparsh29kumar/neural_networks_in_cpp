#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<random>

using namespace std;

double** copy_matrix(double **arr1,int m,int n){
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
	return arr1;
}

double* mat_row_wise_sum(double **arr1,int m,int n){
	double *sum=new double[m];int i,j;
	for(int i=0;i<m;i++){
		sum[i]=0;
		for(j=0;j<n;j++){
			sum[i]+=arr1[i][j];
		}
	}
	return sum;
}

double **transpose(double **arr1,int m,int n){
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
	return arr2;
}

double** softmax(double** arr,int m1,int n1){
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
	return soft;
}

double** mat_element_wise_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2){
	if(m1!=m2||n1!=n2){
		cout<<"matrix dimension mismatch for element wise matrix multiplication"<<endl;
		return NULL;
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
	return mul;
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


double** mat_addition(double** a1,double* a2,int m1,int n1,int m2){
	if(m1!=m2){
		cout<<"matrix dimension mismatch for bias addition"<<endl;
		return NULL;
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
	return add;
}

double** mat_multiply(double** a1,double** a2,int m1,int n1,int m2,int n2){
	if(n1!=m2){
		cout<<"matrix dimension mismatch for matrix multiplication"<<endl;
		return NULL;
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
	return mul;
}

double** relu(double** a1,int m1,int n1){
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
	return act;
}

int input_size = 4;
int hidden_size = 10;
int num_classes = 3;
int num_inputs = 5;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::normal_distribution<double> distribution (0.0,1.0);

class TwoLayerNet{
	pubic:
	double **W1,**b1,**W2,**b2;
	TwoLayerNet(int input_size,int hidden_size,int output_size,float std=1e-4);
}
TwoLayerNet::TwoLayerNet(int input_size,int hidden_size,int output_size,float std){
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
}

TwoLayerNet TwoLayerNet::init_toy_model(){
	return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)
}

void TwoLayerNet::init_toy_data(double **X,int *y){
	X=new double*[num_inputs];
	for(int i=0;i<num_inputs;i++){
		X[i]=new double[input_size];
	}
	y=new int[num_inputs];
	y={0,1,2,2,1};
}
double TwoLayerNet::loss(double reg,double **grads_W1,double **grads_W2,double *grads_b1,double *grads_b2){
    //z1 = X.dot(W1) + b1
    double **z1=mat_addition(mat_multiply(X,W1,num_inputs,input_size,input_size,hidden_size),b2,num_inputs,hidden_size,hidden_size);
    //a1 = np.maximum(0, z1) # pass through ReLU activation function
    double **a1=relu(z1,num_inputs,hidden_size);
    // scores = a1.dot(W2) + b2
    double **score=mat_addition(mat_multiply(a1,W2,num_inputs,hidden_size,hidden_size,output_size),b2,num_inputs,hidden_size,output_size);

    double **probs=softmax(score,num_inputs,output_size);
    //corect_logprobs = -np.log(probs[range(N), y])
    
    double *corect_logprobs=new double[num_inputs];
    
    for(i=0;i<num_inputs;i++){
    	corect_logprobs[i]=-log(probs[i][y[i]]);
    }//now shape of corect_logprobs=number of inputs;
    
    //data_loss = np.sum(corect_logprobs) / N
    double data_loss=sum(corect_logprobs,num_inputs)/num_inputs;
    
    //reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    double reg_loss=0.5*reg*mat_sum(mat_element_wise_multiply(W1,W1,input_size,hidden_size,input_size,hidden_size),input_size,hidden_size)+
    			0.5*reg*mat_sum(mat_element_wise_multiply(W2,W2,hidden_size,output_size,hidden_size,output_size),hidden_size,output_size);
    
    double loss_f = data_loss + reg_loss;
    
    //Backward pass: compute gradients
    
    //dscores = probs
    //dscores[range(N),y] -= 1
    //dscores /= N
    double **dscores=copy_matrix(probs,num_inputs,output_size);
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
    grads_W2=mat_multiply(transpose(a1,num_inputs,hidden_units),dscores,hidden_units,num_inputs,num_inputs,output_size);
    grads_b2=mat_row_wise_sum(dscores,num_inputs,output_size);
    //dhidden = np.dot(dscores, W2.T)
    double **dhidden=mat_multiply(dscores,transpose(W2,hidden_size,output_size),num_inputs,output_size,output_size,hidden_size);
    //backprop the ReLU non-linearity
    for(i=0;i<num_inputs;i++){
    	for(j=0;j<hidden_units;j++){
    		if(a1[i][j]<=0){
    			dhidden[i][j]=0;
    		}
    	}
    }
    //finally into W,b
    //grads['W1'] = np.dot(X.T, dhidden)
    //grads['b1'] = np.sum(dhidden, axis=0)
    grads_W1=mat_multiply(transpose(X,num_inputs,input_size),dhidden,num_inputs,input_size,input_size,hidden_units);
    grads_b1=mat_row_wise_sum(dhidden,num_inputs,hidden_units);
    
    //add regularization gradient contribution
    //grads['W2'] += reg * W2
    //grads['W1'] += reg * W1
    for(i=0;i<hidden_units;i++){
    	for(j=0;j<output_size;j++){
    		grads_W2[i][j]+=W2[i][j]*reg;
    	}
    }
    for(i=0;i<num_inputs;i++){
    	for(j=0;j<hidden_units;j++){
    		grads_W1[i][j]+=W1[i][j]*reg;
    	}
    }
    
    return loss_f;
}

int main(){
	double **X;
	int *y;
	double **grads_W1,**grads_W2,*grads_b1,*grads_b2;
	TwoLayerNet net(input_size, hidden_size, num_classes,1e-1);
	net.init_toy_data(X,y);
	double loss=net.loss(0.1,grads_W1,grads_W2,grads_W2,grads_b1,grads_b2);
	double correct_loss = 1.30378789133;
	cout<<"difference between your loss and correct loss is::"<<loss-correct_loss;
	
	return 0;
}
