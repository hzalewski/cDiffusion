#ifndef DIFFUSION_CORE_H
#define DIFFUSION_CORE_H

void compute_gaussian_kernel(double* data, double* dist, int n, int p, double sigma);
void normalize_markov(double* dist, int n);

#endif