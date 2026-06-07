#' @useDynLib cDiffusion, .registration = TRUE
#' @keywords internal
"_PACKAGE"
#' @title Run diffusion map
#'
#' @description
#' Performs non-linear dimensionality reduction using the exact dense method.
#' Best suited for smaller datasets where calculating the full N x N distance 
#' matrix fits safely into memory.
#'
#' @param data Numeric matrix or data.frame.
#' @param sigma Numeric. Bandwidth for Gaussian kernel (default 1.0).
#' @param k Integer. Number of diffusion dimensions to return (default 2).
#' @param oversampling Integer. Dimensions added for stability in randomized SVD algorithm (default 10). Higher values may negatively impact results of thr algorithm.
#' @param n_iter Integer. Number of subspace iterations in randomized SVD algorithm (default 10).
#' 
#' @return A diffmap object
#' 
#' @examples
#' data <- matrix(runif(100), nrow = 20, ncol = 5)
#' optimal_sigma <- estimate_sigma(data)
#' model <- run_diffusion(data, sigma = optimal_sigma)
#'
#' @export
#' 
#' 

run_diffusion <- function(data, sigma = 1.0, k = 2, oversampling = 20, n_iter = 10) {



    ## maybe default sigma should be estimated instead of 1.0
    ## TODO: better default values for oversampling n_iter and k_neighbours especially for sparse method
    ## TODO: add kmeans after diffusion - easy spectral clustering
    ## Ensure there are not problems when input is too large - also very high-dimensional data 


    if (!is.matrix(data) && !is.data.frame(data)) {
        stop("'data' must be a numeric matrix or a data.frame.")
    }

    # Transpose to be C - contigous
    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    N <- ncol(data_matrix) 

    if (anyNA(data_matrix)) stop("'data' cannot contain NA or NaN values.")
    if (max(data_matrix) == Inf || min(data_matrix) == -Inf) stop("'data' cannot contain Infinite values.")
    
    if (N > 15000) {
        warning(sprintf("WARNING: Number of observations N=%d is too big for 'dense' method. Consider using more memory friendly method 'diffusion_sparse'", N))
    }
    
    m <- as.integer(k + oversampling)

    if (m >= N) {
        warning("WARNING: Overall number of computed dimensions (k + oversampling) is higher than N. Reducing oversampling.")
        m <- as.integer(N - 1)
        if (m <= k) stop("WARNING: N is too small to compute k dimensions.")
    }
    
    res <- .Call("c_run_diffusion", data_matrix, as.numeric(sigma), 
                 m, as.integer(n_iter), PACKAGE="cDiffusion")
    
    # RSVD doesnt guarantee sorted eigenvalues
    idx <- order(res$values, decreasing = TRUE)
    sorted_values <- res$values[idx]
    sorted_vectors <- res$vectors[, idx]
    
    # first vector is trivial so its useless
    diff_coords <- sorted_vectors[, 2:(k+1)]
    diff_vals <- sorted_values[2:(k+1)]
    
    # diffusion map embedding
    for(i in 1:k) {
        diff_coords[, i] <- diff_coords[, i] * diff_vals[i]
    }
    
    result <- list(
        coordinates = diff_coords,
        eigenvalues = diff_vals,
        sigma = sigma,
        data_dim = dim(data),
        method = "dense"
    )
    
    class(result) <- "diffmap"
    return(result)
}


#' @title Run sparse diffusion map
#'
#' @description
#' Performs non-linear dimensionality reduction using a memory-efficient sparse k-NN method.
#' It is better suited for larger datasets than the exact dense method, as its memory 
#' usage scales linearly rather than quadratically.
#' Uses an adaptive Gaussian kernel (local sigma based on the distance to the k-th nearest neighbor).
#'
#' @param data Numeric matrix or data.frame.
#' @param k Integer. Number of diffusion dimensions to return (default 2).
#' @param k_neighbors Integer. Number of nearest neighbors for sparse method (default 15).
#' @param oversampling Integer. Dimensions added for stability in randomized SVD algorithm (default 10). Higher values may negatively impact results of thr algorithm.
#' @param n_iter Integer. Number of subspace iterations in randomized SVD algorithm (default 200).
#' 
#' @return A diffmap object
#' 
#' @examples
#' data <- matrix(runif(5000), nrow = 1000, ncol = 5)
#' model <- run_diffusion_sparse(data, k_neighbors = 20, oversampling = 10, n_iter = 200)
#'
#' @export
run_diffusion_sparse <- function(data, k = 2, k_neighbors = 15, oversampling = 10, n_iter = 200) {

    if (!is.matrix(data) && !is.data.frame(data)) {
        stop("'data' must be a numeric matrix or a data.frame.")
    }

    # Transpose to be C - contigous
    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    N <- ncol(data_matrix) 

    if (anyNA(data_matrix)) stop("'data' cannot contain NA or NaN values.")
    if (max(data_matrix) == Inf || min(data_matrix) == -Inf) stop("'data' cannot contain Infinite values.")
    
    m <- as.integer(k + oversampling)

    if (m >= N) {
        warning("WARNING: Overall number of computed dimensions (k + oversampling) is higher than N. Reducing oversampling.")
        m <- as.integer(N - 1)
        if (m <= k) stop("WARNING: N is too small to compute k dimensions.")
    }
    
    res <- .Call("c_run_sparse_diffusion", data_matrix, 
                 as.integer(k_neighbors), m, as.integer(n_iter), PACKAGE="cDiffusion")
    
    # RSVD doesnt guarantee sorted eigenvalues
    idx <- order(res$values, decreasing = TRUE)
    sorted_values <- res$values[idx]
    sorted_vectors <- res$vectors[, idx]
    
    # first vector is trivial so its useless
    diff_coords <- sorted_vectors[, 2:(k+1)]
    diff_vals <- sorted_values[2:(k+1)]
    
    # diffusion map embedding
    for(i in 1:k) {
        diff_coords[, i] <- diff_coords[, i] * diff_vals[i]
    }
    
    result <- list(
        coordinates = diff_coords,
        eigenvalues = diff_vals,
        data_dim = dim(data),
        method = "sparse"
    )
    
    class(result) <- "diffmap"
    return(result)
}