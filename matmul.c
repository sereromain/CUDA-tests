#include <stdio.h>
#include <stdlib.h>

void matmul(int* mat1, int* mat2, int* out, int dim1, int dim2, int dim_s) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            int o = 0;
            for (int k = 0; k < dim_s; k++) {
                o += mat1[y * dim_s + k] * mat2[k * dim2 + x];
            }
            out[y * dim2 + x] = o;
        }
    }
}

void initialize(int* mat, int dim1, int dim2) {
    for (int y = 0; y < dim1; y++) {
        for (int x = 0; x < dim2; x++) {
            mat[y * dim2 + x] = y * dim2 + x;
        }
    }
}

void printRow(int* row, int size) {
    printf("[");
    for (int x = 0; x < size - 1; x++) {
        printf("%d, ", row[x]);
    }
    if (size != 0) printf("%d", row[size - 1]);
    printf("]\n");
}

void print(int* mat, int dim1, int dim2) {
    printf("[\n");
    for (int y = 0; y < dim1; y++) {
        printf("\t");
        printRow(&mat[y * dim2], dim2);
    }
    printf("]\n");
}

int main(int argc, char** argv) {
    int dim1 = 6;
    int dim_s = 4;
    int dim2 = 5;
    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim_s = (int) atoi(argv[2]);
    if (argc > 3) dim2 = (int) atoi(argv[3]);

    int SIZE1 = dim1 * dim_s;
    int SIZE2 = dim_s * dim2;
    int SIZE = dim1 * dim2;

    int* mat1 = (int*) malloc(SIZE1 * sizeof(int));
    int* mat2 = (int*) malloc(SIZE2 * sizeof(int));
    int* out = (int*) malloc(SIZE * sizeof(int));

    initialize(mat1, dim1, dim_s);
    initialize(mat2, dim_s, dim2);

    matmul(mat1, mat2, out, dim1, dim2, dim_s);

    print(mat1, dim1, dim_s);
    print(mat2, dim_s, dim2);
    print(out, dim1, dim2);

    free(mat1);
    free(mat2);
    free(out);

    return 0;
}