#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;


int main()
{
    srand(time(NULL));
    int t1;
    int *temp = (int*)calloc(sizeof(int),5);
    for(int i = 0;i<3;i++)
    {
        t1 = rand()%5;
        while(temp[t1] == 1)
        {
            t1 = rand()%5;
        }
        temp[t1] = 1;
        printf("%d\n",t1);
    }
}
