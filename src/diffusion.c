#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

SEXP c_run_diffusion(SEXP r_data)
{
    if(!Rf_isReal(r_data)) Rf_error("data must be numeric");
    if(!Rf_isMatrix(r_data)) Rf_error("Data must be a matrix");

    int p = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);

    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n));

    double* data = REAL(r_data);
    double* dist = REAL(r_dist);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i < n; i++){
        dist[i + i*n] = 0.0;

        for(int j = 0; j< i; j++){
                double sum_sq = 0.0;

                for(int k=0; k < p; k++){
                    double diff = data[k + i*p] - data[k + j*p];
                    sum_sq += diff * diff;

                }
                
                double d = sqrt(sum_sq);
                dist[i + j*n] = d;
                dist[j + i*n] = d;


        }

    }

    
    UNPROTECT(1);
    
    return r_dist;
}