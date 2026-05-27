#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include "diffusion_core.h"

SEXP c_run_diffusion(SEXP r_data, SEXP r_sigma, SEXP r_m, SEXP r_n_iter)
{
    int p = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);
    double sigma = Rf_asReal(r_sigma);
    int m = Rf_asInteger(r_m);           
    int n_iter = Rf_asInteger(r_n_iter); 

  
    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n)); 
    SEXP r_X = PROTECT(Rf_allocMatrix(REALSXP, n, m));    
    
  
    SEXP r_D_sqrt = PROTECT(Rf_allocVector(REALSXP, n));  
    
    double* data = REAL(r_data);
    double* dist = REAL(r_dist);
    double* X = REAL(r_X);
    double* D_sqrt = REAL(r_D_sqrt);

  
    apply_gauss(data, dist, n, p, sigma);
    
   
    matrix_normalization(dist, D_sqrt, n);

   
    random_matrix(X, n, m);

    SEXP r_eigvals = PROTECT(Rf_allocVector(REALSXP, m));
    SEXP r_eigvecs = PROTECT(Rf_allocMatrix(REALSXP, n, m));

    randomized_svd(dist, X, REAL(r_eigvecs), REAL(r_eigvals), n, m, n_iter);


    SEXP r_list = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(r_list, 0, r_eigvals);
    SET_VECTOR_ELT(r_list, 1, r_eigvecs);
    
    SEXP r_names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(r_names, 0, Rf_mkChar("values"));
    SET_STRING_ELT(r_names, 1, Rf_mkChar("vectors"));
    Rf_setAttrib(r_list, R_NamesSymbol, r_names);

    UNPROTECT(7);
    return r_list;
}