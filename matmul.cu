/*
_______________        Author : SERE ROMAIN        _______________
--------------- Multiplication de matrices sur GPU ---------------

Ce code permet la multiplication de 2 matrices pouvant être non carrées.
De plus, il est plus performant que la méthode de multiplication de matrices basique en CUDA
(celle qui consiste à charger, dans chaque Thread, une ligne et une colonne afin de les multiplier).
En effet, ce code charge dans la mémoire __Shared__ du Bloc des portions des deux matrices, ce qui permet de calculer la multiplication en plusieurs itérations,
tout en partageant efficacement la mémoire entre les Threads d’un même Bloc.

Pour exécuter le code, il suffit de taper la commande : 
$ code=matmul && nvcc -arch=sm_CC -o $code.o $code.cu && ./$code.o

Dans "sm_CC" remplacer "CC" par la compute capability de votre GPU : https://developer.nvidia.com/cuda-gpus
Exemple : sm_86 ou sm_89 pour des compute capability de 8.6 ou 8.9

L'affichage sera composé des 2 matrices (à multiplier) ainsi que d'une matrice résultat de la multiplication

Pour cet exemple, le programme crée 2 matrices de taille respectives : 48x64 et 64x32 et renvoie une matrice de taille 48x32


Pour comparer les résultats, quelques lignes de Python dans un terminal permettent de visualiser la matrice 48x32 :

>>> import numpy as np
>>> mat1 = np.arange(48*64).reshape(48,64)
>>> mat2 = np.arange(32*64).reshape(64,32)
>>> mat1@mat2
array([[  2731008,   2733024,   2735040, ...,   2789472,   2791488,
          2793504],
       [  6859776,   6865888,   6872000, ...,   7037024,   7043136,
          7049248],
       [ 10988544,  10998752,  11008960, ...,  11284576,  11294784,
         11304992],
       ...,
       [188525568, 188711904, 188898240, ..., 193929312, 194115648,
        194301984],
       [192654336, 192844768, 193035200, ..., 198176864, 198367296,
        198557728],
       [196783104, 196977632, 197172160, ..., 202424416, 202618944,
        202813472]])
*/

#include <stdlib.h>
#include <stdio.h>

#define PRINTIT true

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Fonction de multiplication de matrices sur GPU NVIDIA
__global__ void d_matmul(int* d_mat1, int* d_mat2, int* d_mat3, int dim1, int dim2, int dim_s)
{
    // Variable permettant de stocker la dimension x du block dans lequel se trouve le thread
    int BLOCK_DIM = blockDim.x;

    // Pour chaque thread on calcul sa position (colonne, ligne) dans la matrice finale dmat3
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int lig = blockIdx.y * blockDim.y + threadIdx.y;

    // Allocation de mémoire pour les matrices partagées
    __shared__ int shared_mat1[16*16]; // taille de la matrice mat1
    __shared__ int shared_mat2[16*16]; // taille de la matrice mat2

    // Variable temporaire qui permet d'accumuler les résultats des multiplications du thread
    int tmp = 0;

    // On itère sur toute la matrice le long de la dimension qui est commune aux 2 matrices
    // On saute d'un block à chaque itération
    for (int i = 0; i < dim_s; i += BLOCK_DIM) {

        // Chargement des éléments de la tuile dans la mémoire partagée des threads 
        // Chaque thread va contenir un élément de la matrice 1 et 2 dans la mémoire partagée
        // Tous les threads peuvent donc grâce à cela lire les éléments prélevés par les autres threads
        // Il n'y a donc pas de doublon de mémoire ce qui accélère les calculs
        shared_mat1[threadIdx.y * BLOCK_DIM + threadIdx.x] = d_mat1[lig * dim_s + i + threadIdx.x]; // élément de la tuile i dans la matrice dmat1
        shared_mat2[threadIdx.x * BLOCK_DIM + threadIdx.y] = d_mat2[col + (i + threadIdx.y)*dim2]; // élément de la tuile i dans la matrice dmat2 en transposant le résultat

        // On attend que tous les threads aient finit ce travail avant de passer à la suite des opérations
        __syncthreads();

        // Boucle permettant de faire la multiplication des sous-matrices chargées dans la mémoire partagée et d'ajouter le résultat à l'accumulateur tmp
        for (int j = 0; j < BLOCK_DIM; j++) {
            tmp += shared_mat1[threadIdx.y * BLOCK_DIM + j] * shared_mat2[threadIdx.x * BLOCK_DIM + j]; // multiplication de l'élément j de la ligne de chaque matrice shared (la deuxième  est déjà transposée)
        }

        // On attend que tous les threads aient finit ce travail avant de passer à la suite des opérations   
        __syncthreads();
    }

    // Ecriture du résultat de la multiplication dans la matrice finale dmat3
    d_mat3[lig * dim2 + col] = tmp;
}

// Fonction de multiplication de matrices sur CPU
void h_matmul(int* mat1, int* mat2, int* out, int dim1, int dim2, int dim_s) {
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

void matmul(int* mat1, int* mat2, int* mat3, int dim1, int dim2, int dim_s) {

    int* deviceCount = (int*) malloc(sizeof(int));

    cudaGetDeviceCount(deviceCount);

    if (*deviceCount == 0) {

        h_matmul(mat1, mat2, mat3, dim1, dim2, dim_s);

    } else {
        
        int BLOCK_DIM = 16;

        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
        dim3 blocksPerGrid(
            (dim2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (dim1 + threadsPerBlock.y - 1) / threadsPerBlock.y
        );

        int blockSize = threadsPerBlock.x * threadsPerBlock.y;
        printf(
            "threadsPerBlock.x=%d, threadsPerBlock.y=%d, blocksPerGrid.x=%d, blocksPerGrid.y=%d\n",
            threadsPerBlock.x, threadsPerBlock.y, blocksPerGrid.x, blocksPerGrid.y
        );

        int* d_mat1;
        int* d_mat2;
        int* d_mat3;

        // Allocation de mémoire pour les différentes matrices dans le GPU
        gpuErrchk(cudaMalloc((void**) &d_mat1, dim1 * dim_s * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat2, dim_s * dim2 * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &d_mat3, dim1 * dim2 * sizeof(int)));

        // Copie des 2 matrices dans la mémoire du GPU
        gpuErrchk(cudaMemcpy(d_mat1, mat1, dim1 * dim_s * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_mat2, mat2, dim_s * dim2 * sizeof(int), cudaMemcpyHostToDevice));

        d_matmul<<<blocksPerGrid, threadsPerBlock, blockSize * sizeof(int)>>>(d_mat1, d_mat2, d_mat3, dim1, dim2, dim_s);
        gpuErrchk(cudaPeekAtLastError());

        // Copie du résultat dans la mémoire du CPU
        gpuErrchk(cudaMemcpy(mat3, d_mat3, dim1 * dim2 * sizeof(int), cudaMemcpyDeviceToHost));

        // Libération de la mémoire contenant les matrices
        gpuErrchk(cudaFree(d_mat1));
        gpuErrchk(cudaFree(d_mat2));
        gpuErrchk(cudaFree(d_mat3));
    }
    printf("deviceCount = %d\n", *deviceCount);
    free(deviceCount);
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

    int dim1 = 48;
    int dim2 = 32;
    int dim_s = 64;

    if (argc > 1) dim1 = (int) atoi(argv[1]);
    if (argc > 2) dim2 = (int) atoi(argv[2]);

    int* mat1 = (int*) malloc(dim1 * dim_s * sizeof(int));
    int* mat2 = (int*) malloc(dim_s * dim2 * sizeof(int));
    int* mat3 = (int*) malloc(dim1 * dim2 * sizeof(int));

    initialize(mat1, dim1, dim_s);
    initialize(mat2, dim_s, dim2);

    matmul(mat1, mat2, mat3, dim1, dim2, dim_s);

    if (PRINTIT) {
        printf("Matrice 1 :\n");
        print(mat1, dim1, dim_s);
        printf("Matrice 2 :\n");
        print(mat2, dim_s, dim2);
        printf("Résultat de la multiplication : \n");
        print(mat3, dim1, dim2);
    }

    free(mat1);
    free(mat2);
    free(mat3);

    return 0;
}