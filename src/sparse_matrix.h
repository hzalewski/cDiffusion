#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H


void sparse(double* raw_data, int n, int dim, int k, double sigma, 
                            double** out_data, int** out_indices, int** out_indptr);


void sparse_multiplication(int n, int m, double* data, int* indices, int* indptr, double* X, double* Y);
void sparse_rsvd(double *val, int *col, int *row_ptr, double *X, 
                           double *eigenvectors, double *eigenvalues, int n, int m, int n_iter);

void sparse_normalization(double *val, int *col, int *row_ptr, double *D_sqrt, int n);
void ortogonalize(double *X, int n, int m);
void find_eigenvectors(double *X, double *Y, double *eigenvectors, double *eigenvalues, int n, int m);
#endif 