#include <cuda.h>
#include "mmultiplication.h"

/*
 * TODO
 */
__host__ void gpuMatrixMult(double* A, double* B, double* ANS, long long dim)
{
    /* TODO: Throw an execption if A->columns != B->rows
     */
    long long tempDataSize;
    tempDataSize = dim * dim * dim * sizeof(double);

    double* tempData;
    cudaMalloc((void**)&tempData, tempDataSize);

    long long size;
    size = dim * dim * sizeof(double);

    double* dA;
    cudaMalloc((void**)&dA, size);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);

    double* dB;
    cudaMalloc((void**)&dB, size);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    
    double* dANS;
    cudaMalloc((void**)&dANS, size);

    dim3 blocksPerGrid(dim, dim);
    dim3 threadsPerBlock(dim);
    
    elemMultKernel<<<blocksPerGrid, threadsPerBlock>>>(tempData, dA, dB, dim);
    
    threadsPerBlock = ((dim / 2) + (dim % 2));
    matrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(tempData, dANS, dim);

    cudaMemcpy(ANS, dANS, size, cudaMemcpyDeviceToHost);
}

/*
 * TODO
 */
__global__ void elemMultKernel(double* tempData, double* A, double* B, long long dim)
{
    long long i, j, k;
    j = threadIdx.x;
    k = blockIdx.y;
    i = blockIdx.x;

    if (i < dim && j < dim && k < dim) {
        tempData[j + k * dim + i * dim * dim] = A[i + j * dim] * B[j + k * dim];
    }
    __syncthreads();
}

/*
 * TODO
 */
__global__ void matrixMultKernel(double* tempData, double* ANS, long long dim) 
{
    long long i, j, k, index;
    j = threadIdx.x;
    k = blockIdx.y;
    i = blockIdx.x;

    for (long long stride = (dim / 2); stride >= 1; stride /= 2) {
        index = j + k * dim + i * dim * dim;
        tempData[index] += tempData[index + stride];
    }
    __syncthreads();

    if(!j) {
        ANS[i + k * dim] = tempData[j];
    }
}
