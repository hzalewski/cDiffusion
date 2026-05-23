#include "diffusion_core.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>

void compute_gaussian_kernel(double* data, double* dist, int n, int p, double sigma) {
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < n; i++) {
        dist[i + i*n] = 1.0;

        for(int j = 0; j < i; j++) {
            double sum_sq = 0.0;

            for(int k = 0; k < p; k++) {
                double diff = data[k + i*p] - data[k + j*p];
                sum_sq += diff * diff;
            }
            
            double d = -1.0 * sum_sq / (2.0 * sigma * sigma);
            double w = exp(d);
            
            dist[i + j*n] = w;
            dist[j + i*n] = w;
        }
    }
}

void normalize_markov(double* dist, int n) {

    double* sums = (double*) malloc(n * sizeof(double));
    
    #pragma omp parallel for 
    for(int j = 0; j < n; j++) {
        double column_sums = 0.0;
        
        for(int i = 0; i < n; i++) {
            column_sums += dist[i + j*n];
        }
        sums[j] = column_sums;
    }

    #pragma omp parallel for
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < n; i++) {
            dist[i + j*n] = dist[i + j*n] / sums[i];
        }
    }


    free(sums);
}