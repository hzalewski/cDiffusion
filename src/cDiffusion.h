#ifndef CDIFFUSION_H
#define CDIFFUSION_H

#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Rmath.h>
#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif
#include <Rconfig.h>
#include <R_ext/Lapack.h>



// Node structure for Max-heap 
typedef struct {
    int idx;
    double dist;
} knn_node;



// Helper functions
void random_matrix(double *X, int n, int m);
void ortogonalize(double *X, int n, int m);
void find_eigenvectors(double *X, double *Y, double *eigenvectors, double *eigenvalues, int n, int m);
double sq_dist(double *data, int n, int dim, int m1, int m2);

// Max-heap functions
void down_heap(knn_node* heap, int i, int k);
void build_heap(knn_node* heap, int k);

// Regular matrix (dense) 
void apply_gauss(double *data, double *dist, int n, int p, double sigma);
void matrix_normalization(double *dist, double *D_sqrt, int n);
void matrix_multiplication(double* A, double* X, double* Y, int n, int m);
void randomized_svd(double *A, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter);

// Sparse matrix functions
void sparse(double *data, int n, int dim, int k_neighbors, double sigma, double **out_csr_data, int **out_csr_indices, int **out_csr_indptr);
void sparse_multiplication(int n, int m, double *csr_data, int *csr_indices, int *csr_indptr, double *X, double *Y);
void sparse_normalization(double *csr_data, int *csr_indices, int *csr_indptr, double *D_sqrt, int n);
void sparse_rsvd(double *csr_data, int *csr_indices, int *csr_indptr, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter);
#endif 