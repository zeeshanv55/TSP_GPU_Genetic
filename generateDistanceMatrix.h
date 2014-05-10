void generateDistanceMatrix()
{
	FILE *file;
	int i, j;
	
	srand(time(NULL));

	for (i=0; i<SIZE; i++)
    	{  
		for (j=i; j<SIZE; j++)
		{
			if (i != j)
				distanceMatrix[i*SIZE + j] = distanceMatrix[j*SIZE + i] = rand()%89 + 10;
			else
				distanceMatrix[i*SIZE + j] = 0;
		}
    	}

	file = fopen ("DM.txt", "w");
	for (i=0; i<SIZE; i++)
    	{  
		for (j=0; j<SIZE-1; j++)
			fprintf (file, "%d ", distanceMatrix[i*SIZE + j]);
		fprintf (file, "%d\n", distanceMatrix[(i*SIZE) + (SIZE-1)]);      
    	}
	fclose (file);
}
