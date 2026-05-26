#ifndef DIFFUSION_CORE_H
#define DIFFUSION_CORE_H

void apply_gauss(double *data, double *dist, int n, int p, double sigma);
void matrix_normalization(double *dist, int n);
void mvp(double *matrix, double *vector, double *result, int n);
int eigen(double *matrix, int n, double *eigenvalues, double *eigenvectors);

#endif