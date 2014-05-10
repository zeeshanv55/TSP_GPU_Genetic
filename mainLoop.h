void runGeneticIterations_dynamicMutation()
{
	int globalMinimumFitness = INT_MAX;
	int oldGlobalMinimumFitness;
	double mutationProbability = 0;

	generateRandomPopulation(0);
	
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
	}

	fclose(file);
}

void runGeneticIterations_staticMutation()
{
	int globalMinimumFitness = INT_MAX;
	int oldGlobalMinimumFitness;
	double mutationProbability = 0;

	generateRandomPopulation(0);
	
	FILE *file;
	file = fopen ("Result.txt", "w");
	
	for (int i=0; i<ITERATIONS; i++)
	{
		oldGlobalMinimumFitness = globalMinimumFitness;
		globalMinimumFitness = detectFitIndividuals();
		fprintf(file, "%d, %d\n", i, globalMinimumFitness);
		if (globalMinimumFitness < oldGlobalMinimumFitness)
			saveBestIndividual();
		generateNewGeneration(mutationProbability);
		printf("\n%d Iterations Finished. Minimum fitness reached = %d.", i+1, globalMinimumFitness);
	}

	fclose(file);
}


