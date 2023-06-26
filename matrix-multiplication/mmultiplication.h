#ifndef __MMULTIPLICATION_H__
#define __MMULTIPLICATION_H__
__host__ void gpuMatrixMult(double*, double*, double*, long long);
__global__ void elemMultKernel(double*, double*, double*, long long);
__global__ void matrixMultKernel(double*, double*, long long); 
#endif
