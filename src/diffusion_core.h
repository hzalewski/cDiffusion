#ifndef DIFFUSION_CORE_H
#define DIFFUSION_CORE_H

void compute_gaussian_kernel(double* data, double* dist, int n, int p, double sigma);
void normalize_matrix(double* dist, int n);
void multiply_matrix_vector(double* matrix, double* vector, double* result, int n);


#endif