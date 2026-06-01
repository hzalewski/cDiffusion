#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include "diffusion_core.h"
#include "sparse_matrix.h"
#include "knn.h"

SEXP c_run_diffusion(SEXP r_data, SEXP r_sigma, SEXP r_m, SEXP r_n_iter)
{
    int p = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);
    double sigma = Rf_asReal(r_sigma);
    int m = Rf_asInteger(r_m);
    int n_iter = Rf_asInteger(r_n_iter);

    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    SEXP r_D_sqrt = PROTECT(Rf_allocVector(REALSXP, n));

    double *data = REAL(r_data);
    double *dist = REAL(r_dist);
    double *D_sqrt = REAL(r_D_sqrt);


    //create similirity matrix
    apply_gauss(data, dist, n, p, sigma);
    matrix_normalization(dist, D_sqrt, n);

    SEXP r_eigvals = PROTECT(Rf_allocVector(REALSXP, m));
    SEXP r_eigvecs = PROTECT(Rf_allocMatrix(REALSXP, n, m));
    double *eigvecs = REAL(r_eigvecs);


    
    double *X = (double *)malloc(n * m * sizeof(double));

    //create a random matrix and perform RSVD with it
    random_matrix(X, n, m);
    randomized_svd(dist, X, eigvecs, REAL(r_eigvals), n, m, n_iter);
    free(X);


    // D^-1/2 * v
#pragma omp parallel for
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            eigvecs[i + j * n] /= D_sqrt[i];
    }

    SEXP r_list = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(r_list, 0, r_eigvals);
    SET_VECTOR_ELT(r_list, 1, r_eigvecs);

    SEXP r_names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(r_names, 0, Rf_mkChar("values"));
    SET_STRING_ELT(r_names, 1, Rf_mkChar("vectors"));
    Rf_setAttrib(r_list, R_NamesSymbol, r_names);

    UNPROTECT(6);
    return r_list;
}

SEXP c_run_sparse_diffusion(SEXP r_data, SEXP r_sigma, SEXP r_k_neighbors, SEXP r_m, SEXP r_n_iter)
{
    int dim = Rf_nrows(r_data);
    int n = Rf_ncols(r_data);
    double sigma = Rf_asReal(r_sigma);
    int k_neighbors = Rf_asInteger(r_k_neighbors);
    int m = Rf_asInteger(r_m);
    int n_iter = Rf_asInteger(r_n_iter);

    SEXP r_D_sqrt = PROTECT(Rf_allocVector(REALSXP, n));
    SEXP r_eigvals = PROTECT(Rf_allocVector(REALSXP, m));
    SEXP r_eigvecs = PROTECT(Rf_allocMatrix(REALSXP, n, m));

    double *data = REAL(r_data);
    double *D_sqrt = REAL(r_D_sqrt);
    double *eigvecs = REAL(r_eigvecs);

    //create sparse KNN heap

    double *val = NULL;
    int *col = NULL;
    int *row_ptr = NULL;

    sparse(data, n, dim, k_neighbors, sigma, &val, &col, &row_ptr);
    sparse_normalization(val, col, row_ptr, D_sqrt, n);

    // random matrix and RSVD
    double *X = (double *)malloc(n * m * sizeof(double));
    random_matrix(X, n, m);

    sparse_rsvd(val, col, row_ptr, X, eigvecs, REAL(r_eigvals), n, m, n_iter);


    //D^-1/2 * v
#pragma omp parallel for
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            eigvecs[i + j * n] /= D_sqrt[i];
    }

    free(val);
    free(col);
    free(row_ptr);
    free(X);

    SEXP r_list = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(r_list, 0, r_eigvals);
    SET_VECTOR_ELT(r_list, 1, r_eigvecs);

    SEXP r_names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(r_names, 0, Rf_mkChar("values"));
    SET_STRING_ELT(r_names, 1, Rf_mkChar("vectors"));
    Rf_setAttrib(r_list, R_NamesSymbol, r_names);

    UNPROTECT(5);
    return r_list;
}