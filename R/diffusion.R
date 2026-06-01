#' @useDynLib cDiffusion, .registration = TRUE
#' @keywords internal
"_PACKAGE"

#' @title Run diffusion map
#'
#' @description
#' Main package function to perform non-linear dimensionality reduction.
#' Features automatic method selection between dense exact method 
#' and more memory-efficient sparse k-NN method based on dataset size.
#'
#' @param data Numeric matrix.
#' @param sigma Numeric. Bandwidth for Gaussian kernel (default 1.0).
#' @param k Integer. Number of diffusion dimensions to return (default 2).
#' @param method Character. One of "auto", "dense", "sparse".
#' @param k_neighbors Integer. Number of nearest neighbors for sparse method (default 15).
#' @param oversampling Integer. Dimensions added for stability in randomized SVD algorithm (default 20).
#' @param n_iter Integer. Number of subspace iterations in randomized SVD algorithm (default 10).
#' 
#' @return A diffmap object
#'
#' @export
run_diffusion <- function(data, sigma = 1.0, k = 2, method = "auto", k_neighbors = 15, oversampling = 20, n_iter = 10) {

## maybe default sigma should be estimated instead of 1.0
    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    N <- ncol(data_matrix) 
    
    if (method == "auto") {
        if (N > 2500) {
            method <- "sparse"
            message("N > 2500. Switching to sparse method.")
        } else {
            method <- "dense"
        }
    }
    
    m <- as.integer(k + oversampling)
    
   
    if (method == "sparse") {
        res <- .Call("c_run_sparse_diffusion", data_matrix, as.numeric(sigma), 
                     as.integer(k_neighbors), m, as.integer(n_iter), PACKAGE="cDiffusion")
    } else {
        res <- .Call("c_run_diffusion", data_matrix, as.numeric(sigma), 
                     m, as.integer(n_iter), PACKAGE="cDiffusion")
    }
    
    
    idx <- order(res$values, decreasing = TRUE)
    sorted_values <- res$values[idx]
    sorted_vectors <- res$vectors[, idx]
    
    # first vector is trivial so its useless
    diff_coords <- sorted_vectors[, 2:(k+1)]
    diff_vals <- sorted_values[2:(k+1)]
    
   
    for(i in 1:k) {
        diff_coords[, i] <- diff_coords[, i] * diff_vals[i]
    }
    
    result <- list(
        coordinates = diff_coords,
        eigenvalues = diff_vals,
        sigma = sigma,
        data_dim = dim(data),
        engine_used = method
    )
    
    class(result) <- "diffmap"
    return(result)
}