#ifndef __GNDIFF_H__
#define __GNDIFF_H__

#define BLOCK_SIZE 1

typedef struct 
{
    int width;
    int height;
    int stride;
    int *elements;
} Matrix;

typedef struct 
{
    int gdiff;
    int elem;
    int neighbor;
} Gndiff;

__host__ void GNDiff(const Matrix, Gndiff*);
__device__ void getGNDiff(const Matrix, int, int, Gndiff*);
__global__ void GNDiffKernel(const Matrix, Gndiff*, Gndiff*);
__device__ int getElement(const Matrix, int, int);
__device__ void setElement(Matrix, int, int, int);
__device__ Matrix getSubMatrix(Matrix, int, int);
#endif
