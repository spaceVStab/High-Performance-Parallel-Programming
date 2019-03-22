#include "headers.h"

int main(void){
	int blockSize =  32*32*1;
	int gridSize = 4*2*2;
	int datasize = blockSize * gridSize;

	cudaError_t err = cudaSuccess;
	size_t size = datasize * sizeof(float);

	float *h_I1 = (float *)malloc(size);
	float *h_I2 = (float *)malloc(size);
	float *h_O = (float *)malloc(size);

	//init original data
	if(h_I1 == NULL || h_I2 == NULL || h_O == NULL){
		fprintf(stderr, "Failed to allocate host vectors\n");
		exit(EXIT_FAILURE);
	}

	for(int i=0;i<datasize;i++){
		h_I1[i] = rand()/(float)RAND_MAX;
		h_I2[i] = rand()/(float)RAND_MAX;
	}

	// Allocate the device input vector A
    float *d_I1 = NULL;
    err = cudaMalloc((void **)&d_I1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector I1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_I2 = NULL;
    err = cudaMalloc((void **)&d_I2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector I2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_O = NULL;
    err = cudaMalloc((void **)&d_O, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector O (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_I1, h_I1, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector I1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_I2, h_I2, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector I2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //running kernel 1:
    dim3 grid(4,2,2);
    dim3 block(32,32,1);
    process_kernel1<<<grid, block>>>(d_I1, d_I2, d_O, datasize);

   	//copy the device output to host output
   	cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

   	//free previous inputs; since of no use now
   	err = cudaFree(d_I1);
   	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector I1 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

   	err = cudaFree(d_I2);
   	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector I2 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_I1);
	free(h_I2);


	// allocate device memory for output of kernel_2
	float *d_O2 = NULL;
    err = cudaMalloc((void **)&d_O2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector O2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid2(2,8,1);
    dim3 block2(8,8,16);
    process_kernel2<<<grid2, block2>>>(d_O, d_O2, datasize);

    //copy the device output to host output
   	cudaMemcpy(h_O, d_O2, size, cudaMemcpyDeviceToHost);

   	err = cudaFree(d_O);
   	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector O (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// allocate device memory for output of kernel_2
	float *d_O3 = NULL;
    err = cudaMalloc((void **)&d_O3, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector O3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid3(8,1,1);
    dim3 block3(128,16,1);
    process_kernel3<<<grid3, block3>>>(d_O3, d_O2, datasize);

    //copy the device output to host output
   	cudaMemcpy(h_O, d_O3, size, cudaMemcpyDeviceToHost);

   	err = cudaFree(d_O2);
   	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector O2 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_O3);
   	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector O3 (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	free(h_O);
	return 0;

}