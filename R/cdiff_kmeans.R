#' @title Spectral Clustering using Diffusion Maps with Eigengap Heuristic
#'
#' @description
#' Applies K-means clustering on the diffusion coordinates. If the number of 
#' clusters is not provided, it automatically estimates the optimal number of 
#' clusters (k) using the Eigengap heuristic (finding the largest drop between 
#' consecutive eigenvalues).
#'
#' @param x A 'diffmap' object returned by `run_diffusion()` or `run_diffusion_sparse()`.
#' @param centers Integer or "auto". The number of clusters. If "auto" (default), optimal k is estimated.
#' @param max_k Integer. Maximum number of clusters to consider when `centers = "auto"` (default 10).
#' @param nstart Integer. Number of random sets to choose for K-means (default 25 for stability).
#'
#' @return A standard 'kmeans' list object containing the clustering results.
#' 
#' @importFrom stats kmeans
#' @export
cdiff_kmeans <- function(x, centers = "auto", max_k = 10, nstart = 25) {
  
  if (!inherits(x, "diffmap")) {
    stop("ERROR: 'x' must be an object of class 'diffmap'.")
  }
  
  eigenvalues <- x$eigenvalues
  coords <- x$coordinates
  
  if (is.character(centers) && centers == "auto") {
    
    if (length(eigenvalues) < 3) {
      stop("ERROR: Not enough eigenvalues to compute eigengap. Run diffusion map with a higher 'dims' argument.")
    }
    
    limit <- min(max_k, length(eigenvalues))
    gaps <- abs(diff(eigenvalues[1:limit]))
    optimal_k <- which.max(gaps) + 1
    
    if (optimal_k < 2) {
      optimal_k <- 2
      message("Eigengap suggested 1 cluster. Changing to k = 2.")
    } else {
      
      message(sprintf("Eigengap heuristic: selected k = %d clusters.", optimal_k))
    }
    
  } else if (is.numeric(centers) && length(centers) == 1 && centers >= 2) {
    optimal_k <- as.integer(centers)
  } else {
    stop("ERROR: 'centers' must be 'auto' or a single integer >= 2.")
  }
  
  dims<- min(optimal_k, ncol(coords))
  X <- coords[, 1:dims, drop = FALSE]
  

  res <- kmeans(X, centers = optimal_k, nstart = nstart)
  
  return(res)
}