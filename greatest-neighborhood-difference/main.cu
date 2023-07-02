#include <stdio.h>
#include <stdlib.h>
#include "gndiff.h"

int main(void)
{
    Matrix A;

    scanf("%d", &A.height);
    A.stride = (A.width = A.height);

    size_t size;
    size = A.width * A.height * sizeof(int);

    cudaMallocHost((void**)&A.elements, size);
    for (int i = 0; i < A.height; ++i) {
        for(int j = 0; j < A.width; ++j) {
            scanf("%d", &A.elements[j + i * A.width]);
        }
    }

    Gndiff gndiff;
    GNDiff(A, &gndiff);
    cudaDeviceSynchronize();
    
    int first, second; 
    if (gndiff.elem > gndiff.neighbor) {
        first = gndiff.elem;
        second = gndiff.neighbor;
    }
    else {
        first = gndiff.neighbor;
        second = gndiff.elem;
    }
    printf("%d %d", first, second);
    
    return 0;
}
