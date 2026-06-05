#include "cDiffusion.h"

SEXP c_run_diffusion(SEXP r_data, SEXP r_sigma, SEXP r_m, SEXP r_n_iter)
{
  
    if (!Rf_isReal(r_data)) Rf_error("data must be numeric");
    if (!Rf_isMatrix(r_data)) Rf_error("data must be a matrix");

    int dim = Rf_nrows(r_data); 
    int n = Rf_ncols(r_data); 

    if (!Rf_isReal(r_sigma)) Rf_error("sigma must be numeric");
    if (Rf_length(r_sigma) != 1) Rf_error("sigma must be a scalar");
    double sigma = Rf_asReal(r_sigma);
    if (sigma <= 0.0) Rf_error("sigma must be greater than zero");

    if (!Rf_isInteger(r_m) && !Rf_isReal(r_m)) Rf_error("number of dimensions (m) must be numeric");
    int m = Rf_asInteger(r_m);
    if (m < 1) Rf_error("number of dimensions (m) must be greater than zero");
    if (m >= n) Rf_error("number of dimensions (m) must be strictly less than the number of samples.");

    if (!Rf_isInteger(r_n_iter) && !Rf_isReal(r_n_iter)) Rf_error("n_iter must be numeric");
    int n_iter = Rf_asInteger(r_n_iter);
    if (n_iter < 1) Rf_error("n_iter must be greater than zero");
   

    SEXP r_dist = PROTECT(Rf_allocMatrix(REALSXP, n, n));
    SEXP r_D_sqrt = PROTECT(Rf_allocVector(REALSXP, n));

    double *data = REAL(r_data);
    double *dist = REAL(r_dist);
    double *D_sqrt = REAL(r_D_sqrt);

    apply_gauss(data, dist, n, dim, sigma);
    matrix_normalization(dist, D_sqrt, n);

    SEXP r_eigvals = PROTECT(Rf_allocVector(REALSXP, m));
    SEXP r_eigvecs = PROTECT(Rf_allocMatrix(REALSXP, n, m));
    double *eigvecs = REAL(r_eigvecs);

    double *X = malloc(n * m * sizeof(double));
    random_matrix(X, n, m);
    randomized_svd(dist, X, eigvecs, REAL(r_eigvals), n, m, n_iter);
    free(X);

    // Rescale eigenvectors using inverse of degree matrix
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
    if (!Rf_isReal(r_data)) Rf_error("data must be numeric");
    if (!Rf_isMatrix(r_data)) Rf_error("data must be a matrix");
    
    int dim = Rf_nrows(r_data); 
    int n = Rf_ncols(r_data);   

    if (!Rf_isReal(r_sigma)) Rf_error("sigma must be numeric");
    if (Rf_length(r_sigma) != 1) Rf_error("sigma must be a scalar");
    double sigma = Rf_asReal(r_sigma);
    if (sigma <= 0.0) Rf_error("sigma must be greater than zero");

    if (!Rf_isInteger(r_k_neighbors) && !Rf_isReal(r_k_neighbors)) Rf_error("k_neighbors must be numeric");
    int k_neighbors = Rf_asInteger(r_k_neighbors);
    if (k_neighbors < 1) Rf_error("k_neighbors must be greater than zero");

    if (!Rf_isInteger(r_m) && !Rf_isReal(r_m)) Rf_error("number of dimensions (m) must be numeric");
    int m = Rf_asInteger(r_m);
    if (m < 1) Rf_error("number of dimensions (m) must be greater than zero");
    if (m >= n) Rf_error("number of dimensions (m) must be strictly less than the number of samples.");

    if (!Rf_isInteger(r_n_iter) && !Rf_isReal(r_n_iter)) Rf_error("n_iter must be numeric");
    int n_iter = Rf_asInteger(r_n_iter);
    if (n_iter < 1) Rf_error("n_iter must be greater than zero");
    if (k_neighbors >= n) Rf_error("k_neighbors must be strictly less than the number of samples.");
  

    SEXP r_D_sqrt = PROTECT(Rf_allocVector(REALSXP, n));
    SEXP r_eigvals = PROTECT(Rf_allocVector(REALSXP, m));
    SEXP r_eigvecs = PROTECT(Rf_allocMatrix(REALSXP, n, m));

    double *data = REAL(r_data);
    double *D_sqrt = REAL(r_D_sqrt);
    double *eigvecs = REAL(r_eigvecs);

    double *csr_data = NULL;
    int *csr_indices = NULL;
    int *csr_indptr = NULL;

    sparse(data, n, dim, k_neighbors, sigma, &csr_data, &csr_indices, &csr_indptr);

  
    sparse_normalization(csr_data, csr_indices, csr_indptr, D_sqrt, n);

    double *X = malloc(n * m * sizeof(double));
    random_matrix(X, n, m);

    sparse_rsvd(csr_data, csr_indices, csr_indptr, X, eigvecs, REAL(r_eigvals), n, m, n_iter);

    // Re-scale eigenvectors using inverse of degree matrix
    #pragma omp parallel for
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
            eigvecs[i + j * n] /= D_sqrt[i];
    }


    free(csr_data);
    free(csr_indices);
    free(csr_indptr);
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