#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include "diffusion_core.h"

SEXP c_run_diffusion(SEXP r_data, SEXP r_sigma)
{
    
    if(!Rf_isReal(r_data)) Rf_error("data must be numeric");
    if(!Rf_isMatrix(r_data)) Rf_error("Data must be a matrix");

    if(!Rf_isReal(r_sigma)) Rf_error("sigma must be numeric.");
    if(Rf_length(r_sigma) != 1) Rf_error("sigma must be a scalar.");

    double sigma = Rf_asReal(r_sigma);
    if(sigma <= 0.0) Rf_error("sigma must be greater than zero");

    
    int p = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);

    
    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    double* data = REAL(r_data);
    double* dist = REAL(r_dist);
    

    compute_gaussian_kernel(data, dist, n, p, sigma);

    normalize_markov(dist, n);
    
    UNPROTECT(1);
    return r_dist;
}