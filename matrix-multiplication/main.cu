#include <stdio.h>
#include <stdlib.h>
#include "mmultiplication.h"

int main(int argc, char** argv)
{
    long long dim;
    dim = atoi(argv[1]);

    double* A;
    A = (double*)malloc(dim * dim * sizeof(double));

    double* B;
    B = (double*)malloc(dim * dim * sizeof(double));

    for (long long i = 0; i < dim; ++i) {
        for (long long j = 0; j < dim; ++j) {
            A[j + i * dim] = 1.0;
            B[j + i * dim] = 1.0 /* ((double)dim)*/;
        }
    }

    double* ANS;
    ANS = (double*)malloc(dim * dim * sizeof(double));
    gpuMatrixMult(A, B, ANS, dim);

    for (long long i = 0; i < dim; ++i) {
        for (long long j = 0; j < dim; ++j) {
            printf("%.2lf\t", ANS[j + i * dim]);
        }
        printf("\n");
    }

    return 0;
}
