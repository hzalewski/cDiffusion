#' @useDynLib cDiffusion, .registration = TRUE
#' @keywords internal
"_PACKAGE"


#' @title Run diffusion map
#'
#' @description
#' Main package function to perform non-linear dimensionality reduction using 
#' Randomized Singular Value Decomposition (rSVD) and Markov matrix normalization.
#'
#' @param data Numeric matrix containing the input data.
#' @param sigma Numeric scalar for the Gaussian kernel bandwidth (default 1.0).
#' @param k Integer. Number of diffusion dimensions to return (default 2).
#' @param oversampling Integer. Parameter for Randomized SVD algorithm which states how many additional dimensions are used, improving accuracy of results and numerical stability (default 100).
#' @param n_iter Integer. Number of subspace iterations in Randomized SVD (default 10).
#' 
#' @return
#' Returns an object of S3 class \code{diffmap}, containing:
#' \itemize{
#'   \item \code{coordinates} - A numeric matrix of the reduced dimensions.
#'   \item \code{eigenvalues} - A numeric vector of eigenvalues.
#'   \item \code{sigma} - The Gaussian kernel parameter used.
#'   \item \code{data_dim} - Dimensions of the original input data.
#' }
#'
#' @export
run_diffusion <- function(data, sigma = 1.0, k = 2, oversampling = 5, n_iter = 100) {

    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    
    
    m <- as.integer(k + oversampling)
    
  
    res <- .Call("c_run_diffusion", data_matrix, as.numeric(sigma), m, as.integer(n_iter), PACKAGE="cDiffusion")
    

    idx <- order(res$values, decreasing = TRUE)
    sorted_values <- res$values[idx]
    sorted_vectors <- res$vectors[, idx]
    
   
    diff_coords <- sorted_vectors[, 2:(k+1)] 
    diff_vals <- sorted_values[2:(k+1)]

    v1 <- abs(sorted_vectors[, 1])
    D_inv_sqrt <- 1 / v1
    
    for(i in 1:k) {
        diff_coords[, i] <- (diff_coords[, i] * D_inv_sqrt) * diff_vals[i]
    }

  
    
    result <- list(
        coordinates = diff_coords,
        eigenvalues = diff_vals,
        sigma = sigma,
        data_dim = dim(data)
    )
    
    class(result) <- "diffmap"

    
    return(result)
}