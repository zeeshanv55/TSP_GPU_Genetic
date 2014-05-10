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

