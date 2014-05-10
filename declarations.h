#define SIZE 50  //Number of cities to travel
#define POPULATION_SIZE 32640 //Number of individuals in a population (FIT_POPULATION_SIZE C 2)
#define FIT_POPULATION_SIZE 256 //Number of individuals selected to breed for the next generation
#define ITERATIONS 256 //Number of iterations till the final result
#define BLOCK_NUMBER 32 //Number of blocks in a cuda-grid
#define THREAD_NUMBER 1020 //Number of threads in a cuda-block (BLOCK_NUMBER*THREAD_NUMBER = POPULATION_SIZE)

int distanceMatrix[SIZE*SIZE];  //The distance matrix containing distances between all cities
int population[POPULATION_SIZE*SIZE];  //The genetic population matrix with each row as an individual tour
int fitPopulation[FIT_POPULATION_SIZE*SIZE];  //The fittest individuals detected in a population
int populationFitnessVector[POPULATION_SIZE];  //The vector containing the fitness values of the corresponding rows in the population matrix
int minimumFitnessIndicesVector[FIT_POPULATION_SIZE];  //The vector containing the indices of the minimum fitness value individuals of a population
int randomMutationVector[POPULATION_SIZE]; //The vector containing random values to help decide for mutation of child
int bestIndividual[SIZE]; //The best chromosome found
int bestIndividualFitness = INT_MAX; //Fitness of bestIndividual

__global__ void cross(int*, int*, int*, int*, int*, int*, int*, double);
void generateRandomPopulation(int);
int detectFitIndividuals();
void generateNewGeneration(double);
int saveBestIndividual();
void runGeneticIterations_dynamicMutation();
void runGeneticIterations_staticMutation();
void generateDistanceMatrix();
