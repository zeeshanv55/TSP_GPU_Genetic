To run: 
1) set suitable parameters in 'declarations.h'. Follow comments in the file.
2) nvcc kernel.cu
3) ./a.out

Note:
1) You need an NVIDIA GPU and nvcc compiler (CUDA compiler) to run this.
2) Use runGeneticIterations_staticMutation() in 'kernel.cu' for a constant value of mutation probability.
3) Use runGeneticIterations_dynamicMutation() in 'kernel.cu' for a varying values of mutation probability.
