#include <cuda.h>
#include <stdio.h>
#include "thresholding.h"

__host__ void imgThresholding(Matrix Img, int threshold)
{
   Matrix devImg;
   devImg.width = Img.width;
   devImg.height = Img.height;
   devImg.stride = Img.stride;

   size_t size;
   size = Img.width * Img.height * sizeof(int);

   cudaMalloc((void**)&devImg.elements, size);
   cudaMemcpy(devImg.elements, Img.elements, size, cudaMemcpyHostToDevice);

   dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
   dim3 dimGrid(Img.width / BLOCK_WIDTH + (Img.width % BLOCK_WIDTH),\
           Img.height / BLOCK_HEIGHT + (Img.height % BLOCK_HEIGHT));

   thresholdingKernel<<<dimGrid, dimBlock>>>(devImg, threshold);

   cudaMemcpy(Img.elements, devImg.elements, size, cudaMemcpyDeviceToHost);

} 
__device__ int getElement(const Matrix matrix, int row, int col)
{
    return matrix.elements[row * matrix.stride + col];
}

__device__ void setElement(Matrix matrix, int row, int col,
        int value)
{
    matrix.elements[row * matrix.stride + col] = value;
}

__device__ Matrix getSubMatrix(Matrix matrix, int row, int col)
{
    Matrix subMatrix;
    subMatrix.width = BLOCK_WIDTH;
    subMatrix.height = BLOCK_HEIGHT;
    subMatrix.stride = matrix.stride;
    subMatrix.elements = &matrix.elements[matrix.stride * BLOCK_HEIGHT * row
        + BLOCK_WIDTH * col];
    return subMatrix;
}

__global__ void thresholdingKernel(Matrix Img, int threshold)
{
    int blockRow, blockCol;
    blockRow = blockIdx.y;
    blockCol = blockIdx.x;

    Matrix subImg;
    subImg = getSubMatrix(Img, blockRow, blockCol);

    int row, col;
    row = threadIdx.y;
    col = threadIdx.x;
    
    //__shared__ sharedSubImg[BLOCK_WIDTH][BLOCK_HEIGHT];
    //sharedSubImg[row][col] = getElement(subImg, row, col);
    int elem;
    elem = getElement(subImg, row, col);
    setElement(subImg, row, col, (elem > threshold) ? 1 : 0);
}
