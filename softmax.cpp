#include<iostream>
#include<math.h>
#include<stdlib.h>

using namespace std;

float* softmax(int arr[],int n){
	int sum=0,i;
	float *sm=new float[n];
	for(i=0;i<n;i++){
		sum+=exp(arr[i]);
	}
	for(i=0;i<n;i++){
		sm[i]=exp(arr[i])/sum;
	}
	return sm;
}

int main(){
	int n,i;
	int *arr=NULL;
	cout<<"enter n::";
	cin>>n;
	arr=new int[n];
	for(i=0;i<n;i++){
		cin>>arr[i];
	}
	float *soft=softmax(arr,n);
	float sum=0;
	for(i=0;i<n;i++){
		cout<<soft[i]<<endl;
		sum+=soft[i];
	}
	cout<<"sum="<<sum;
}
