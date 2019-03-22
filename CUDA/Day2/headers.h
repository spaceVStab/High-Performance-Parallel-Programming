#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void process_kernel1( float *input1, float *input2, float *output, int datasize);
__global__ void process_kernel2( float *input, float *output, int datasize);
__global__ void process_kernel3( float *input, float *output, int datasize);