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
run_diffusion <- function(data, sigma = 1.0, k = 2) {

    data_matrix <- as.matrix(data)
    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    res <- .Call("c_run_diffusion", data_matrix, as.numeric(sigma), as.integer(k), PACKAGE="cDiffusion")

    idx <- order(res$values, decreasing = TRUE)
    sorted_values <- res$values[idx]
    sorted_vectors <- res$vectors[, idx]


    diff_coords <- sorted_vectors[, 2:(k+1)]
    diff_vals <- sorted_values[2:(k+1)]
    
    
    result <- list(
        coordinates = diff_coords,
        eigenvalues = diff_vals,
        sigma = sigma,
        data_dim = dim(data)
    )
    
    class(result) <- "diffmap"
    return(result)
}