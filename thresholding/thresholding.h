#ifndef __THRESHOLDING_H__
#define __THRESHOLDING_H__

#define BLOCK_WIDTH 3
#define BLOCK_HEIGHT 3

typedef struct 
{
    int width;
    int height;
    int stride;
    int *elements;
} Matrix;

__host__ void imgThresholding(Matrix, int);
__device__ int getElement(const Matrix, int, int);
__device__ void setElement(Matrix, int, int, int);
__device__ Matrix getSubMatrix(Matrix, int, int);
__global__ void thresholdingKernel(Matrix, int);
#endif
