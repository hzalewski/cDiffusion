#' @useDynLib cDiffusion
"_PACKAGE"

#' @title
#' Run diffusion map
#'
#' @description
#' Main package function.
#'
#' @param data numeric matrix
#' @param sigma numeric scalar for Gaussian kernel (default 1.0)
#'
#' @return
#' returns a numeric matrix
#'
#' @export
run_diffusion <- function(data, sigma = 1.0, k = 2, oversampling = 5, n_iter = 300) {

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