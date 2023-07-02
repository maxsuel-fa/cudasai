#include<stdio.h>
#include "gndiff.h"

/*
* TODO
*/
__host__ void GNDiff(const Matrix matrix, Gndiff* gndiff)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    size_t gridSize;
    gridSize = matrix.width / BLOCK_SIZE + (matrix.width % BLOCK_SIZE > 1);
    dim3 dimGrid(gridSize, gridSize);

    size_t inSize;
    inSize = matrix.width * matrix.height * sizeof(Gndiff);

    Gndiff* inGndiffList;
    cudaMallocHost((void**)&inGndiffList, inSize);

    Gndiff* devInGndiffList;
    cudaMalloc((void**)&devInGndiffList, inSize);

    size_t outSize;
    outSize = gridSize * gridSize * sizeof(Gndiff);

    Gndiff* outGndiffList;
    cudaMallocHost((void**)&outGndiffList, outSize);

    Gndiff* devOutGndiffList;
    cudaMalloc((void**)&devOutGndiffList, outSize);

    Matrix devMatrix;
    devMatrix.width = matrix.width;
    devMatrix.height = matrix.height;
    devMatrix.stride = matrix.stride;
    size_t size;
    size = devMatrix.width * devMatrix.height * sizeof(int);
    cudaMalloc((void**)&devMatrix.elements, size);
    cudaMemcpy(devMatrix.elements, matrix.elements, size, cudaMemcpyHostToDevice);

    GNDiffKernel<<<dimGrid, dimBlock>>>(devMatrix, devInGndiffList, devOutGndiffList);

    cudaMemcpy(inGndiffList, devInGndiffList, inSize, cudaMemcpyDeviceToHost);

    gndiff->gdiff = INT_MIN;
    inSize = matrix.width * matrix.height;

    for (int i = 0; i < inSize; ++i) {
        if ((inGndiffList[i]).gdiff > gndiff->gdiff) {
            gndiff->gdiff = (inGndiffList[i]).gdiff;
            gndiff->elem = (inGndiffList[i]).elem;
            gndiff->neighbor = (inGndiffList[i]).neighbor;
        }
    }

}

/*
 * TODO
 */
__device__ int getElement(const Matrix matrix, int row, int col)
{
    return matrix.elements[row * matrix.stride + col];
}

/* 
* TODO
*/
__device__ void setElement(Matrix matrix, int row, int col,
        int value)
{
    matrix.elements[row * matrix.stride + col] = value;
}

/* 
* TODO
*/
__device__ Matrix getSubMatrix(Matrix matrix, int row, int col)
{
    Matrix subMatrix;
    subMatrix.width = BLOCK_SIZE;
    subMatrix.height = BLOCK_SIZE;
    subMatrix.stride = matrix.stride;
    subMatrix.elements = &matrix.elements[matrix.stride * BLOCK_SIZE * row
        + BLOCK_SIZE * col];
    return subMatrix;
}

/* 
* TODO
*/
__device__ void getGNDiff(const Matrix matrix, int elemRow, int elemCol,\
        Gndiff* gndiff)
{
    int globalRow, globalCol;
    int diff;

    gndiff->gdiff = INT_MIN;
    for (int i = elemRow - 1; i <= elemRow + 1; ++i) {
        for(int j = elemCol - 1; j <= elemCol + 1; ++j) {
            globalRow = blockIdx.y * blockDim.y + i;
            globalCol = blockIdx.x * blockDim.x + j;
            if ((globalRow >= 0 && globalRow < (matrix.stride))\
                    && (globalCol >= 0 && globalCol < (matrix.stride))\
                    && (i != elemRow || j != elemCol)) {
                diff = getElement(matrix, elemRow, elemCol);
                diff -= getElement(matrix, i, j);
                diff = abs(diff);
                if (diff > gndiff->gdiff) {
                    gndiff->gdiff = diff;
                    gndiff->elem = getElement(matrix, elemRow, elemCol);
                    gndiff->neighbor = getElement(matrix, i, j);
                }
            }
        }
    }
}

/* 
* TODO
*/
__global__ void GNDiffKernel(const Matrix matrix, Gndiff* inGndiffList, Gndiff* outGndiffList)
{
    int elemRow, elemCol;
    elemRow = threadIdx.y;
    elemCol = threadIdx.x;

    int blockRow, blockCol;
    blockRow = blockIdx.y;
    blockCol = blockIdx.x;

    int globalThreadRow, globalThreadCol;
    globalThreadRow = blockRow * blockDim.y + elemRow;
    globalThreadCol = blockCol * blockDim.x + elemCol;

    if (globalThreadRow < matrix.height && globalThreadCol < matrix.width) {
        Matrix subMatrix;
        subMatrix = getSubMatrix(matrix, blockRow, blockCol);
        int index;
        index = globalThreadRow * matrix.stride + globalThreadCol;
        getGNDiff(subMatrix, elemRow, elemCol, &inGndiffList[index]);
    }
}
