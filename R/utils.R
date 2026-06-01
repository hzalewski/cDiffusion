#' @title Estimate optimal sigma for Gaussian kernel
#'
#' @description
#' Uses the median heuristic to automatically estimate the optimal bandwidth 
#' (sigma) parameter for the Gaussian kernel in diffusion maps.
#'
#' @param data Numeric matrix of input data.
#' @param sample_size Integer. Number of points to sample for estimation (default 500).
#'
#' @return A numeric scalar representing the estimated sigma.
#' @export
estimate_sigma <- function(data, sample_size = 500) {
  N <- nrow(data)
  

  if (N <= sample_size) {
    sub_data <- data
  } else {
  
    idx <- sample(1:N, sample_size)
    sub_data <- data[idx, ]
  }
  

  dists <- as.matrix(dist(sub_data))
  

  upper_tri_dists <- dists[upper.tri(dists)]
  
 
  optimal_sigma <- median(upper_tri_dists) / sqrt(2)
  
  return(optimal_sigma)
}