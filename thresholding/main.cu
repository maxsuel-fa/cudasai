#include <stdio.h>
#include <stdlib.h>
#include "thresholding.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("The path of the input file was not provided...\n");
        exit(EXIT_FAILURE);
    }

    FILE* fp;
    fp = fopen(argv[1], "r");
    if (!fp) {
        printf("Error in opening the file\n");
        exit(EXIT_FAILURE);
    }

    Matrix Img;

    fscanf(fp, "%d %d", &Img.height, &Img.width);
    Img.stride = Img.width;

    int threshold;
    fscanf(fp, "%d", &threshold);

    size_t size;
    size = Img.width * Img.height * sizeof(int);

    cudaMallocHost((void**)&Img.elements, size);
    for (int i = 0; i < Img.height; ++i) {
        for(int j = 0; j < Img.width; ++j) {
            fscanf(fp, "%d", &Img.elements[j + i * Img.width]);
        }
    }

    imgThresholding(Img, threshold);
    cudaDeviceSynchronize();

    for (int i = 0; i < Img.height; ++i) {
        for(int j = 0; j < Img.width; ++j) {
            printf("%d ", Img.elements[j + i * Img.width]);
        }
        printf("\n");
    }

    return 0;
}
