
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define SIZE 50  //Number of cities to travel
#define POPULATION_SIZE 294528 //Number of individuals in a population (FIT_POPULATION_SIZE C 2)
#define FIT_POPULATION_SIZE 768 //Number of individuals selected to breed for the next generation
#define ITERATIONS 128 //Number of iterations till the final result
#define BLOCK_NUMBER 384 //Number of blocks in a cuda-grid
#define THREAD_NUMBER 767 //Number of threads in a cuda-block (BLOCK_NUMBER*THREAD_NUMBER = POPULATION_SIZE)

int distanceMatrix[SIZE*SIZE];  //The distance matrix containing distances between all cities
int population[POPULATION_SIZE*SIZE];  //The genetic population matrix with each row as an individual tour
int fitPopulation[FIT_POPULATION_SIZE*SIZE];  //The fittest individuals detected in a population
int populationFitnessVector[POPULATION_SIZE];  //The vector containing the fitness values of the corresponding rows in the population matrix
int minimumFitnessIndicesVector[FIT_POPULATION_SIZE];  //The vector containing the indices of the minimum fitness value individuals of a population
int randomMutationVector[POPULATION_SIZE]; //The vector containing random values to help decide for mutation of child
int bestIndividual[SIZE];
int bestIndividualFitness = INT_MAX;

__global__ void cross(int dev_fitPopulation[], int dev_population[], int dev_populationFitnessVector[], int dev_distanceMatrix[], int dev_randomMutationVector[], int dev_randomMutationPoint1[], int dev_randomMutationPoint2[], double mutationProbability)
{
	int I, J, i, j, k, flag = 0, counter = 0, threadNumber = (blockIdx.x*blockDim.x + threadIdx.x);

	for (I=0; I<FIT_POPULATION_SIZE; I++)
	{
		for (J=I+1; J<FIT_POPULATION_SIZE; J++)
		{
			if (counter == threadNumber)
				{
					flag=1;
					break;
				}
			else
				counter++;
		}
		
		if (flag==1)
			break;
	}

	int mother[SIZE];
	int father[SIZE];
	int child1[SIZE];
	int child2[SIZE];
	int child[SIZE];
	int child_m[SIZE];
	int index_check[SIZE];

	for (int k=0; k<SIZE; k++)
	{
		mother[k] = dev_fitPopulation[I*SIZE + k];
		father[k] = dev_fitPopulation[J*SIZE + k];
	}

	for (k=0; k<SIZE; k++)
		index_check[k] = 0;

	child1[0] = mother[0];
	index_check[child1[0]] = 1;
	i=1;
	j=0;
	k=1;
	while (k < SIZE)
	{
		if (index_check[mother[i]] == 1)
		{
			i++;
			i=i%SIZE;
			continue;
		}

		if (index_check[father[j]] == 1)
		{
			j++;
			j=j%SIZE;
			continue;
		}

		if (dev_distanceMatrix[child1[k-1]*SIZE + mother[i]] <= dev_distanceMatrix[child1[k-1]*SIZE + father[j]])
		{
			child1[k] = mother[i];
			k++;
			index_check[mother[i]] = 1;
		}
		else
		{
			child1[k] = father[j];
			k++;
			index_check[father[j]] = 1;
		}
	}

	for (k=0; k<SIZE; k++)
		index_check[k] = 0;

	child2[0] = father[0];
	index_check[child2[0]] = 1;
	i=0;
	j=1;
	k=1;
	while (k < SIZE)
	{
		if (index_check[mother[i]] == 1)
		{
			i++;
			i=i%SIZE;
			continue;
		}

		if (index_check[father[j]] == 1)
		{
			j++;
			j=j%SIZE;
			continue;
		}

		if (dev_distanceMatrix[child2[k-1]*SIZE + mother[i]] <= dev_distanceMatrix[child2[k-1]*SIZE + father[j]])
		{
			child2[k] = mother[i];
			k++;
			index_check[mother[i]] = 1;
		}
		else
		{
			child2[k] = father[j];
			k++;
			index_check[father[j]] = 1;
		}
	}

	int fitnessMother = 0;
	for (k=0; k<SIZE-1; k++)
		fitnessMother += dev_distanceMatrix[mother[k]*SIZE + mother[k+1]];
	fitnessMother += dev_distanceMatrix[mother[0]*SIZE + mother[SIZE-1]];

	int fitnessFather = 0;
	for (k=0; k<SIZE-1; k++)
		fitnessFather += dev_distanceMatrix[father[k]*SIZE + father[k+1]];
	fitnessFather += dev_distanceMatrix[father[0]*SIZE + father[SIZE-1]];

	int fitnessChild1 = 0;
	for (k=0; k<SIZE-1; k++)
		fitnessChild1 += dev_distanceMatrix[child1[k]*SIZE + child1[k+1]];
	fitnessChild1 += dev_distanceMatrix[child1[0]*SIZE + child1[SIZE-1]];

	int fitnessChild2 = 0;
	for (k=0; k<SIZE-1; k++)
		fitnessChild2 += dev_distanceMatrix[child2[k]*SIZE + child2[k+1]];
	fitnessChild2 += dev_distanceMatrix[child2[0]*SIZE + child2[SIZE-1]];

	int fitnessChild;
	if (fitnessChild1 < fitnessChild2)
	{
		for (k=0; k<SIZE; k++)
		{
			child[k] = child1[k];
		}
		fitnessChild = fitnessChild1;
	}
	else
	{
		for (k=0; k<SIZE; k++)
		{
			child[k] = child2[k];
		}
		fitnessChild = fitnessChild2;
	}

	if (dev_randomMutationVector[threadNumber] < mutationProbability)
	{
		for (k=0; k<SIZE; k++)
			child_m[k] = child[k];

		child_m[dev_randomMutationPoint1[threadNumber]] = child[dev_randomMutationPoint2[threadNumber]];
		child_m[dev_randomMutationPoint2[threadNumber]] = child[dev_randomMutationPoint1[threadNumber]];

		int fitness_child_m = 0;
		for (k=0; k<SIZE-1; k++)
			fitness_child_m += dev_distanceMatrix[child_m[k]*SIZE + child_m[k+1]];
		fitness_child_m += dev_distanceMatrix[child_m[0]*SIZE + child_m[SIZE-1]];

		if(fitness_child_m < fitnessChild)
		{
			fitnessChild = fitness_child_m;
			for (k=0; k<SIZE; k++)
				child[k] = child_m[k];
		}
	}

	if (fitnessChild < fitnessMother && fitnessChild < fitnessFather)
	{
		dev_populationFitnessVector[threadNumber] = fitnessChild;
		for (k=0; k<SIZE; k++)
			dev_population[threadNumber*SIZE + k] = child[k];
	}
	else if (fitnessMother < fitnessFather && fitnessMother < fitnessChild)
	{
		dev_populationFitnessVector[threadNumber] = fitnessMother;
		for (k=0; k<SIZE; k++)
			dev_population[threadNumber*SIZE + k] = mother[k];
	}
	else
	{
		dev_populationFitnessVector[threadNumber] = fitnessFather;
		for (k=0; k<SIZE; k++)
			dev_population[threadNumber*SIZE + k] = father[k];
	}
}

void generateRandomPopulation(int preservePrevious = 0)
{
	if (preservePrevious == 0)
	{
		for (int i=0; i<POPULATION_SIZE; i++)
		{
			for (int j=0; j<SIZE; j++)
			{
				int flag;
				int t;

				do {
					flag = 0;
					t = rand()%SIZE;
					for (int k=0; k<j; k++)
					{
						if (t == population[i*SIZE + k])
						{
							flag = 1;
							break;
						}
					}
				}while(flag == 1);

				population[i*SIZE + j] = t;
			}

			populationFitnessVector[i] = 0;
			for (int j=0; j<SIZE-1; j++)
				populationFitnessVector[i] += distanceMatrix[ (population[i*SIZE + j]*SIZE + population[i*SIZE + j + 1]) ];
			populationFitnessVector[i] += distanceMatrix[ (population[i*SIZE]*SIZE + population[i*SIZE + SIZE - 1]) ];
		}
	}
	else
	{
		for (int j=0; j<SIZE; j++)
			population[j] = fitPopulation[j];

		populationFitnessVector[0] = 0;
		for (int j=0; j<SIZE-1; j++)
			populationFitnessVector[0] += distanceMatrix[ (population[j]*SIZE + population[j + 1]) ];
		populationFitnessVector[0] += distanceMatrix[ (population[0]*SIZE + population[SIZE - 1]) ];

		for (int i=1; i<POPULATION_SIZE; i++)
		{
			for (int j=0; j<SIZE; j++)
			{
				int flag;
				int t;

				do {
					flag = 0;
					t = rand()%SIZE;
					for (int k=0; k<j; k++)
					{
						if (t == population[i*SIZE + k])
						{
							flag = 1;
							break;
						}
					}
				}while(flag == 1);

				population[i*SIZE + j] = t;
			}

			populationFitnessVector[i] = 0;
			for (int j=0; j<SIZE-1; j++)
				populationFitnessVector[i] += distanceMatrix[ (population[i*SIZE + j]*SIZE + population[i*SIZE + j + 1]) ];
			populationFitnessVector[i] += distanceMatrix[ (population[i*SIZE]*SIZE + population[i*SIZE + SIZE - 1]) ];
		}
	}
}

int detectFitIndividuals()
{
	int minimumFitness, minimumFitnessIndex, populationFitnessVector_Copy[POPULATION_SIZE], globalMinimumFitness;

	for (int i=0; i<POPULATION_SIZE; i++)
		populationFitnessVector_Copy[i] = populationFitnessVector[i];

	for (int i=0; i<FIT_POPULATION_SIZE; i++)
	{
		minimumFitness = INT_MAX;
		minimumFitnessIndex = -1;
		for (int j=0; j<POPULATION_SIZE; j++)
		{
			if (populationFitnessVector_Copy[j] < minimumFitness)
			{
				minimumFitness = populationFitnessVector_Copy[j];
				minimumFitnessIndex = j;
			}
		}
		
		for (int j=0; j<POPULATION_SIZE; j++)
		{
			if (populationFitnessVector_Copy[j] == populationFitnessVector_Copy[minimumFitnessIndex])
			{
				populationFitnessVector_Copy[j] = INT_MAX;
			}
		}

		for (int j=0; j<SIZE; j++)
		{
			fitPopulation[i*SIZE + j] = population[minimumFitnessIndex*SIZE + j];
		}

		if (i==0)
		{
			globalMinimumFitness = minimumFitness;
		}
	}

	return (globalMinimumFitness);
}

void generateNewGeneration(double mutationProbability)
{
	int randomMutationPoint1[POPULATION_SIZE];
	int randomMutationPoint2[POPULATION_SIZE];

	int *dev_population = 0;
	int *dev_populationFitnessVector = 0;
	int *dev_distanceMatrix = 0;
	int *dev_fitPopulation = 0;
	int *dev_randomMutationVector = 0;
	int *dev_randomMutationPoint1 = 0;
	int *dev_randomMutationPoint2 = 0;

	for (int i=0; i<POPULATION_SIZE; i++)
	{
		randomMutationVector[i] = rand()%100;
		randomMutationPoint1[i] = rand()%SIZE;
		while (randomMutationPoint1[i] == (randomMutationPoint2[i] = rand()%SIZE));
	}

	cudaMalloc((void**)&dev_population, SIZE * POPULATION_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_populationFitnessVector, POPULATION_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_fitPopulation, SIZE * FIT_POPULATION_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_distanceMatrix, SIZE * SIZE * sizeof(int));
	cudaMalloc((void**)&dev_randomMutationVector, POPULATION_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_randomMutationPoint1, POPULATION_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_randomMutationPoint2, POPULATION_SIZE * sizeof(int));
	
	cudaMemcpy(dev_fitPopulation, fitPopulation, SIZE * FIT_POPULATION_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_distanceMatrix, distanceMatrix, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randomMutationVector, randomMutationVector, POPULATION_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randomMutationPoint1, randomMutationPoint1, POPULATION_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randomMutationPoint2, randomMutationPoint2, POPULATION_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(THREAD_NUMBER);
	dim3 numBlocks(BLOCK_NUMBER);
	cross<<<numBlocks,threadsPerBlock>>>(dev_fitPopulation, dev_population, dev_populationFitnessVector, dev_distanceMatrix, dev_randomMutationVector, dev_randomMutationPoint1, dev_randomMutationPoint2, mutationProbability);

	cudaMemcpy(population, dev_population, SIZE * POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(populationFitnessVector, dev_populationFitnessVector, POPULATION_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_population);
	cudaFree(dev_populationFitnessVector);
	cudaFree(dev_distanceMatrix);
	cudaFree(dev_fitPopulation);
	cudaFree(dev_randomMutationVector);
	cudaFree(dev_randomMutationPoint1);
	cudaFree(dev_randomMutationPoint2);
}

int saveBestIndividual()
{
	int minimumFitness=INT_MAX, minimumFitnessIndex, populationFitnessVector_copy[POPULATION_SIZE], tempBestIndividual[SIZE];

	for (int i=0; i<POPULATION_SIZE; i++)
		populationFitnessVector_copy[i] = populationFitnessVector[i];

	for (int j=0; j<POPULATION_SIZE; j++)
	{
		if (populationFitnessVector_copy[j] < minimumFitness)
		{
			minimumFitness = populationFitnessVector_copy[j];
			minimumFitnessIndex = j;
		}
	}

	int tempBestIndividualFitness = 0;
	for (int i=0; i<SIZE; i++)
		tempBestIndividual[i] = population[SIZE*minimumFitnessIndex + i];

	for (int i=0; i<SIZE-1; i++)
		tempBestIndividualFitness += distanceMatrix[tempBestIndividual[i]*SIZE + tempBestIndividual[i+1]];
	tempBestIndividualFitness += distanceMatrix[tempBestIndividual[0]*SIZE + tempBestIndividual[SIZE-1]];

	if (tempBestIndividualFitness < bestIndividualFitness)
	{
		bestIndividualFitness = tempBestIndividualFitness;
		for (int i=0; i<SIZE; i++)
			bestIndividual[i] = tempBestIndividual[i];
		return (1);
	}
	else
		return (0);
}

void runGeneticIterations()
{
	srand(time(NULL));
	int globalMinimumFitness = INT_MAX;
	int oldGlobalMinimumFitness;
	double mutationProbability = 0;

	generateRandomPopulation();
	
	FILE *file;
	file = fopen ("Result.txt", "w");
	
	for (int i=0; i<ITERATIONS; i++)
	{
		oldGlobalMinimumFitness = globalMinimumFitness;
		globalMinimumFitness = detectFitIndividuals();
		printf("\n%d Iterations Finished. Minimum fitness reached = %d.", i+1, globalMinimumFitness);

		fprintf(file, "%d, %d\n", i, globalMinimumFitness);

		if (oldGlobalMinimumFitness <= globalMinimumFitness)
		{
			mutationProbability += (100.0 - mutationProbability)/1.44;
		}
		else
		{
			mutationProbability = 0.0;
		}

		if (mutationProbability >= 99.0)
		{
			printf("\nPhase Complete. Minimum fitness reached = %d. %d percent completed.\n", globalMinimumFitness, i*100/ITERATIONS);
			globalMinimumFitness = INT_MAX;
			mutationProbability = 0.0;
			generateRandomPopulation(saveBestIndividual());
			i++;
			oldGlobalMinimumFitness = globalMinimumFitness;
			globalMinimumFitness = detectFitIndividuals();
			printf("\n%d Iterations Finished. Minimum fitness reached = %d.", i+1, globalMinimumFitness);
			fprintf(file, "%d, %d\n", i, globalMinimumFitness);
		}

		generateNewGeneration(mutationProbability);

		/*oldGlobalMinimumFitness = globalMinimumFitness;
		globalMinimumFitness = detectFitIndividuals();
		fprintf(file, "%d, %d\n", i, globalMinimumFitness);
		if (globalMinimumFitness < oldGlobalMinimumFitness)
			saveBestIndividual();
		generateNewGeneration(mutationProbability);*/

	}

	fclose(file);
}

int main()
{
	FILE *file;
	int i, j;
	
	file = fopen ("DM.txt", "r");
	for (i=0; i<SIZE; i++)
    	{  
		for (j=0; j<SIZE-1; j++)
			fscanf (file, "%d ", &distanceMatrix[i*SIZE + j]);
		fscanf (file, "%d", &distanceMatrix[(i*SIZE) + (SIZE-1)]);      
    	}
	fclose (file);

	runGeneticIterations();
	saveBestIndividual();
	
	for (i=0; i<SIZE; i++)
		printf("%d ", bestIndividual[i]);
	printf("- %d\n\n", bestIndividualFitness);

	cudaDeviceReset();
    	return 0;
}
