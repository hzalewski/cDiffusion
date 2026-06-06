#' @title Estimate optimal sigma for Gaussian kernel
#'
#' @description
#' Uses the median heuristic to automatically estimate the optimal bandwidth 
#' (sigma) parameter for the Gaussian kernel in diffusion maps.
#'
#' @param data Numeric matrix or data.frame.
#' @param sample_size Integer. Number of points to sample for estimation (default 500).
#'
#' @return A numeric scalar representing the estimated sigma.
#' 
#' @importFrom stats dist median
#' 
#' @examples
#'
#' data <- matrix(rnorm(100), nrow = 20, ncol = 5)
#'
#' optimal_sigma <- estimate_sigma(data)
#' 
#' @export
estimate_sigma <- function(data, sample_size = 500) {
  
  
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("ERROR: 'Data' must be numeric matrix or data.frame object")
  }

  if (!is.numeric(data)) {
    stop("ERROR: Data must be numeric.")
  }
  
  data_matrix <- as.matrix(data)
  
  if (anyNA(data_matrix)) {
    stop("ERROR: Data cannot have missing values.")
  }
  
  if (!is.numeric(sample_size) || sample_size < 2) {
    stop("ERROR: 'sample_size' must be a number > 1.")
  }
  
  N <- nrow(data_matrix)
  if (N < 2) {
    stop("ERROR: Data set must contan at least 2 rows.")
  }
  
 
  if (N <= sample_size) {
    sub_data <- data_matrix
  } else {
    idx <- sample(1:N, sample_size)
    sub_data <- data_matrix[idx, ]
  }

  dists <- as.matrix(dist(sub_data))
  upper_tri_dists <- dists[upper.tri(dists)]
  
  if (length(upper_tri_dists) == 0 || all(upper_tri_dists == 0)) {
    warning("Warning: All points are identical. Returning defualt sigma = 1.0")
    return(1.0)
  }
  
  optimal_sigma <- median(upper_tri_dists) / sqrt(2)
  
  return(optimal_sigma)
}