
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "declarations.h"
#include "generateDistanceMatrix.h"
#include "helpingFunctions.h"
#include "mainLoop.h"

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


int main()
{
	generateDistanceMatrix();

	runGeneticIterations_dynamicMutation();
	saveBestIndividual();
	
	printf("\n\nBest solution reached\n");
	for (int i=0; i<SIZE; i++)
		printf("%d ", bestIndividual[i]);
	printf("- %d\n\n", bestIndividualFitness);

	cudaDeviceReset();
    	return 0;
}
