#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#define NUM_OF_TOTAL_DATA 2850
#define NUM_OF_LEARNING 2000
#define NUM_OF_TEST 850
#define INPUT 8
#define HIDDEN_ONE 8
#define HIDDEN_TWO 8
#define OUTPUT 7
#define ALPHA 0.1
#define EPOCH 1000000
/*hidden layers*/
/***************/
typedef struct Node{
	double sum;
	double activated_sum;
	double* weight;
	double delta;
	struct Node** connection;	
	struct Node** pre_connection;
}NODE;
NODE* inputLayer;
NODE* hiddenLayer_one;
NODE* hiddenLayer_two;
NODE* outputLayer;

void init_node(NODE* node, int pre_size, int size, int con_size, NODE* connection, NODE* pre_connection);
void free_all_node();
void init_model();
int get_learning_data(double learning_data[][INPUT+OUTPUT]);
/* train */
double forward_propagate(double* input_data);
void train_node_output(NODE* layer, double* target);
void train_node_hidden(NODE* node, int size, int con_size, int order);
/* activation functions */
double sigmoid(double u);
double sigmoid_derivative(double u);
double ReLU(double u);
double ReLU_derivative(double u);
void softmax(double* value, int size);
void softmax_derivative(NODE* output, double* target);
double loss_MSE(double* output, double* target);
double loss_cross_entropy(double* output, double* target);
int classificate(double* target, double* output);
void print_node(NODE* node, int size);
double init_rand();

void main(){
	srand((unsigned)time(NULL));
	double learning_data[NUM_OF_TOTAL_DATA][INPUT+OUTPUT];
	double confusionMatrix[OUTPUT][5];
	double loss=0;
	long int i=0;
	get_learning_data(learning_data);
	init_model();
	while(i<EPOCH){
		for(int i=0;i<NUM_OF_LEARNING;i++){
			loss = forward_propagate(learning_data[i]);
	
			train_node_output(outputLayer, learning_data[i]);
			
			for(int j=0;j<HIDDEN_TWO;j++)
				train_node_hidden(&hiddenLayer_two[j],HIDDEN_TWO,OUTPUT,j);
			for(int j=0;j<HIDDEN_ONE;j++)
				train_node_hidden(&hiddenLayer_one[j],HIDDEN_ONE,HIDDEN_TWO,j);
		}
		fflush(stdout);
		printf("epoch : %ld loss : %lf weight : %lf %lf %lf\r", i,loss,outputLayer->weight[0], hiddenLayer_two->weight[0], hiddenLayer_one->weight[0]);
		i++;
	}
	for(int i=0;i<OUTPUT;i++)
		print_node(&outputLayer[i],OUTPUT);
	for(int i=0;i<HIDDEN_ONE;i++)
		print_node(&hiddenLayer_one[i],HIDDEN_ONE);
	for(int i=0;i<HIDDEN_TWO;i++)
		print_node(&hiddenLayer_two[i],HIDDEN_TWO);
	//printf("epoch : %ld err : %lf weight : %lf %lf %lf\r", i,err,outputLayer->weight[1], hiddenLayer_two->weight[1], hiddenLayer_one->weight[1]);
	free_all_node();
}
void init_node(NODE* node, int pre_size, int size, int con_size, NODE* connection, NODE* pre_connection){
	
	node->sum=0; node->activated_sum=0; node->delta=0;

	if(pre_size == 0)
		node->pre_connection = NULL;
	else{
		node->weight = (double*)malloc(sizeof(double)*pre_size);
		node->pre_connection = (NODE**)malloc(sizeof(NODE*)*pre_size);
		for(int i=0;i<pre_size;i++){
			node->pre_connection[i] = &pre_connection[i];
			node->weight[i] = init_rand();
		}
	}
	if(con_size == 0)
		node->connection = NULL;
	else{
		node->connection = (NODE**)malloc(sizeof(NODE*)*con_size);
		for(int i=0;i<con_size;i++)
			node->connection[i] = &connection[i];
	}
}
void free_all_node(){
	free(inputLayer);
	free(hiddenLayer_one);
	free(hiddenLayer_two);
	free(outputLayer);
}
void init_model(){
	/* declare and allocate all layers */
	inputLayer = (NODE*)malloc(sizeof(NODE)*INPUT);
	hiddenLayer_one = (NODE*)malloc(sizeof(NODE)*HIDDEN_ONE);
	hiddenLayer_two = (NODE*)malloc(sizeof(NODE)*HIDDEN_TWO);
	outputLayer = (NODE*)malloc(sizeof(NODE)*OUTPUT);
	/* initiate all layers */
	for(int i=0;i<INPUT;i++)
		init_node(&inputLayer[i],0,INPUT,HIDDEN_ONE,hiddenLayer_one,inputLayer);
	for(int i=0;i<HIDDEN_ONE;i++)
		init_node(&hiddenLayer_one[i],INPUT,HIDDEN_ONE,HIDDEN_TWO,hiddenLayer_two,inputLayer);
	for(int i=0;i<HIDDEN_TWO;i++)
		init_node(&hiddenLayer_two[i],HIDDEN_ONE,HIDDEN_TWO,OUTPUT,outputLayer,hiddenLayer_one);
	for(int i=0;i<OUTPUT;i++)
		init_node(&outputLayer[i],HIDDEN_TWO,OUTPUT,0,hiddenLayer_two,outputLayer);
}
double forward_propagate(double* input_data){
	double tmp[OUTPUT] ={0,};
	double err=0;
	/* input layer */
	for(int i=0;i<INPUT;i++)
		inputLayer[i].activated_sum = input_data[i];
	/* hidden layer stage 1 */
	for(int i=0;i<HIDDEN_ONE;i++){
		hiddenLayer_one[i].sum = 0;
		for(int j=0;j<INPUT;j++)
			hiddenLayer_one[i].sum += hiddenLayer_one[i].weight[j]*inputLayer[j].activated_sum;
		hiddenLayer_one[i].activated_sum = sigmoid(hiddenLayer_one[i].sum);
	} /* hidden layer stage 2 */
	for(int i=0;i<HIDDEN_TWO;i++){
		hiddenLayer_two[i].sum = 0;
		for(int j=0;j<HIDDEN_ONE;j++)
			hiddenLayer_two[i].sum += hiddenLayer_two[i].weight[j]*hiddenLayer_one[j].activated_sum;
		hiddenLayer_two[i].activated_sum = sigmoid(hiddenLayer_two[i].sum);
	}/* output layer*/
	for(int i=0;i<OUTPUT;i++){
		outputLayer[i].sum = 0;
		for(int j=0;j<HIDDEN_TWO;j++)
			outputLayer[i].sum += outputLayer[i].weight[j]*hiddenLayer_two[j].activated_sum;
		tmp[i] = outputLayer[i].sum;	
	}/* softmax */
	softmax(tmp,OUTPUT);
	for(int i=0;i<OUTPUT;i++)
		outputLayer[i].activated_sum = tmp[i];
	/* */
	err=loss_cross_entropy(tmp,input_data);
	return err;
}
void train_node_output(NODE* layer, double* target){
	double delta[OUTPUT]={0,};
	double delta_matrix[OUTPUT][OUTPUT]={{0,},};
	softmax_derivative(layer,target);
	for(int i=0;i<OUTPUT;i++){
		for(int j=0;j<HIDDEN_TWO;j++)
			layer[i].weight[j] -= ALPHA*layer[i].delta*layer[i].pre_connection[j]->activated_sum;
	}
}
void train_node_hidden(NODE* node, int size, int con_size, int order){
	node->delta = 0;
	for(int i=0;i<con_size;i++)
		node->delta += node->connection[i]->delta*node->connection[i]->weight[order]*sigmoid_derivative(node->activated_sum);
	/*update weight*/
	for(int i=0;i<size;i++){
		node->weight[i] -=  ALPHA*node->delta*node->pre_connection[i]->activated_sum;
	}
}
int get_learning_data(double learning_data[][INPUT+OUTPUT]){
	int num_of_data=0;
	int i=0, j=0;
	char buf[64];
	FILE* fp = fopen("data","r");
	if(fp != NULL){
		while(!feof(fp)){
			fscanf(fp,"%s",buf);
			if(j>(INPUT+OUTPUT)-1){
				i++; j=0;	
			}
			learning_data[i][j] = atof(buf);
			j++;
		}
		memset(buf,0,64);
	}
	fclose(fp);
	return num_of_data;
}
double sigmoid(double u)
{
	return 1.0/(1.0+exp(-u));
}
double sigmoid_derivative(double u){
	return (1-u)*u;
}
double ReLU(double u){
	if (u <= 0)
		return 0;
	return u;
}
double ReLU_derivative(double u){
	if(u>0)
		return 1;
	else
		return 0;
}
void softmax(double* value, int size){
	double sum_exp=0;
	for(int i=0;i<size;i++)
		sum_exp += exp(value[i]);
	for(int i=0;i<size;i++)
		value[i] = exp(value[i])/sum_exp;
}
void softmax_derivative(NODE* output, double* target){
	for(int i=0;i<OUTPUT;i++){
		output[i].delta =0;
		for(int j=0;j<OUTPUT;j++){
			if(i=j)
				output[i].delta += output[i].activated_sum*(1-output[i].activated_sum);
			else
				output[i].delta += (-1)*target[i+INPUT]*output[j].activated_sum;
		}
	}
}
double loss_MSE(double* output, double* target){
	double result=0;
	for(int i=0;i<OUTPUT;i++)
		result += (output[i]-target[INPUT+i])*(output[i]-target[INPUT+i])/2; 
	return result;
}
double loss_cross_entropy(double* output, double* target){
	double crossEntropy=0;
	double buf=0;
	for(int i=0;i<OUTPUT;i++){
		buf += output[i];
		crossEntropy += target[i+INPUT]*log(output[i])+(1-target[i+INPUT])*log(1-output[i]);
	}
		crossEntropy = crossEntropy*(-1);
	return crossEntropy;
}
int classificate(double* target, double* output){
	for(int i=0;i<OUTPUT;i++){
		if(output[i] > 0.8 || output[i] > 1.2)
			output[i] == 1;
		else
			output[i] == 0;
	}
	for(int i=0;i<OUTPUT;i++){
		if(target[i] != output[i])
			return 0;
	}
	return 1;
}
void print_node(NODE* node, int size){
	printf("[");
	for(int i=0;i<size;i++)
		printf("%lf,", node->weight[i]);
	printf("]\n");
}
 /*
}
void confusion_matrix(double** confusionMatrix, double** target, double** output){
	int table[OUTPUT][2]={{0,},};
	for(int i=0;i<NUM_OF_TEST;i++){
		for(int j=0;j<OUTPUT;j++){
			if(target[i][j] == 1){
				if(classificate(target[i],output[i]))
					table[j][0] += 1;
				else
					table[j][1] += 1;
			}
		}
	}

}*/
double init_rand(){
 	double random_number;
 	while((random_number=(double)rand()/RAND_MAX)==1.0);
 		random_number=random_number*2-1;
 	return random_number; 
}