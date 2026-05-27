#ifndef DIFFUSION_CORE_H
#define DIFFUSION_CORE_H

void apply_gauss(double *data, double *dist, int n, int p, double sigma);
void matrix_normalization(double *dist, double *D_sqrt, int n);
void mvp(double *matrix, double *vector, double *result, int n);


void random_matrix(double *X, int n, int m);
void randomized_svd(double *A, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter);
void matrix_multiplication(double* A, double* X, double* Y, int n, int m);

#endif