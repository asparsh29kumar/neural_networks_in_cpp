#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<random>

using namespace std;

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::normal_distribution<double> distribution (0.0,1.0);

class TwoLayerNet{
	pubic:
	int **W1,**b1,**W2,**b2;
	TwoLayerNet(int input_size,int hidden_size,int output_size,float std=1e-4);
}
TwoLayerNet::TwoLayerNet(int input_size,int hidden_size,int output_size,float std){
	W1=new int*[input_size];
	b1=new int[hidden_size];
	W2=new int*[hidden_size];
	b2=new int[output_size];
	for(int i=0;i<input_size;i++){
		W1[i]=new int[hidden_size];
	}
	for(int i=0;i<hidden_size;i++){
		W2[i]=new int[output_size];
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

int input_size = 32 * 32 * 3
int hidden_size = 50
int num_classes = 10
int num_inputs = 5

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
void TwoLayerNet::loss(){
# TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).   
    
    score=new int*[N];
    for(int i=0;i<N;i++){
    	score[i]=new int[num_classes];
    }
    for(int i=0;i<N;i++){
    	for(int j=0;j<num_classes;j++){
    		score[i][j]=0;
    	}
    }
    for(int i=0;i<N;)

}

int main(){
	double **X;
	int *y;

	TwoLayerNet net=init_toy_model();
	net.init_toy_data(X,y);
		



	return 0;
}
