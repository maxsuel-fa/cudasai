#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DIM 4
#define BLOCK_SIZE 16

typedef struct
{
    int width, height, stride;
    int* elements;
} Matrix;

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
    subMatrix.width = BLOCK_SIZE;
    subMatrix.height = BLOCK_SIZE;
    subMatrix.stride = matrix.stride;
    subMatrix.elements = &matrix.elements[matrix.stride * BLOCK_SIZE * row
        + BLOCK_SIZE * col];
    return subMatrix;
}

__global__ void transpAndSumKernel(Matrix A, Matrix B, Matrix C)
{
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bRow, bCol, tRow, tCol;

    bRow = blockIdx.y;
    bCol = blockIdx.x;

    Matrix subA, subB, subC;
    subA = getSubMatrix(A, bRow, bCol);
    subB = getSubMatrix(B, bRow, bCol);
    subC = getSubMatrix(C, bRow, bCol);

    tRow = threadIdx.y;
    tCol = threadIdx.x;

    As[tCol][tRow] = getElement(subA, tRow, tCol);
    Bs[tCol][tRow] = getElement(subB, tRow, tCol);

    int value;
    value = As[tCol][tRow] + Bs[tCol][tRow];
    setElement(subC, tRow, tCol, value);
}

int main(void)
{
    Matrix A, B, C;
    A.width = A.height = A.stride = DIM;
    B.width = B.height = B.stride = DIM;
    C.width = C.height = C.stride = DIM;

    size_t size;
    size = DIM * DIM * sizeof(int);

    cudaMallocManaged(&A.elements, size);
    cudaMallocManaged(&B.elements, size);
    cudaMallocManaged(&C.elements, size);

    srand(48);
    for (int i = 0; i < DIM; ++i) {
        for(int j = 0; j < DIM; ++j) {
            A.elements[j + j * DIM] = rand() % DIM;
            B.elements[j + j * DIM] = rand() % DIM;
        }
    }
    
    dim3 bDim(BLOCK_SIZE, BLOCK_SIZE);
    int gSize = A.width / BLOCK_SIZE + (A.width % BLOCK_SIZE != 0);
    dim3 gDim(gSize, gSize);

    transpAndSumKernel<<<gDim, bDim>>>(A, B, C);
    cudaDeviceSynchronize();

    for (int i = 0; i < DIM; ++i) {
        for(int j = 0; j < DIM; ++j) {
            printf("%d ", C.elements[j + i * DIM]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < DIM; ++i) {
        for(int j = 0; j < DIM; ++j) {
            printf("%d ", A.elements[j + i * DIM]);
        }
        printf("\n");
    }
    printf("\n");
    for (int i = 0; i < DIM; ++i) {
        for(int j = 0; j < DIM; ++j) {
            printf("%d ", B.elements[j + i * DIM]);
        }
        printf("\n");
    }

    return 0;
}
