#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define NUM_OF_TOTAL_DATA 2850
#define NUM_OF_LEARNING 2000
#define NUM_OF_TEST 850
#define INPUT 8
#define HIDDEN_ONE 8
#define HIDDEN_TWO 8
#define OUTPUT 7
#define ALPHA 0.0001
#define EPOCH 1000

int main(){

	double data[NUM_OF_TOTAL_DATA][INPUT+OUTPUT];
	int num_of_data=0;
	int i=0, j=0;
	char buf[64];
	FILE* fp = fopen("data","r");
	 if (fp == NULL)
    {
        fprintf(stderr, "File Open Error!\n");
        exit(1);
    }

	if(fp != NULL){
		while(!feof(fp)){
			fscanf(fp,"%s",buf);
			if(j>(INPUT+OUTPUT)-1){
				i++; j=0;	
			}
			data[i][j] = atof(buf);
			j++;
		}
		memset(buf,0,64);
	}
	for(int i=0;i<20;i++){
		for(int j=0;j<15;j++)
			printf("%lf  ",data[i][j]);
		printf("\n");
	}
	fclose(fp);
	return num_of_data;
}