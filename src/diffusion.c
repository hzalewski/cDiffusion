#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include "diffusion_core.h"

SEXP c_run_diffusion(SEXP r_data, SEXP r_sigma, SEXP r_k)
{

    if (!Rf_isReal(r_data))
        Rf_error("data must be numeric");
    if (!Rf_isMatrix(r_data))
        Rf_error("Data must be a matrix");

    if (!Rf_isReal(r_sigma))
        Rf_error("sigma must be numeric");
    if (Rf_length(r_sigma) != 1)
        Rf_error("sigma must be a scalar");

    if (!Rf_isInteger(r_k) && !Rf_isReal(r_k))
        Rf_error("number of dimensions (k) must be numeric");
    int k = Rf_asInteger(r_k);
    if (k < 1)
        Rf_error("number of dimensions (k) must be greater than zero");

    double sigma = Rf_asReal(r_sigma);
    if (sigma <= 0.0)
        Rf_error("sigma must be greater than zero");

    int p = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);

    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    double *data = REAL(r_data);
    double *dist = REAL(r_dist);

    apply_gauss(data, dist, n, p, sigma);

    matrix_normalization(dist, n);

    SEXP r_eigenvalues = PROTECT(Rf_allocVector(REALSXP, n));
    SEXP r_eigenvectors = PROTECT(Rf_allocMatrix(REALSXP, n, n));

    int info = eigen(dist, n, REAL(r_eigenvalues ), REAL(r_eigenvectors));

    if (info != 0)
    {
        Rf_error("LAPACK error: %d", info);
    }

    SEXP r_list = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(r_list, 0, r_eigenvalues);
    SET_VECTOR_ELT(r_list, 1, r_eigenvectors);

    SEXP r_names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(r_names, 0, Rf_mkChar("values"));
    SET_STRING_ELT(r_names, 1, Rf_mkChar("vectors"));
    Rf_setAttrib(r_list, R_NamesSymbol, r_names);

    UNPROTECT(5);
    return r_list;
}
