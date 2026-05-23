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
run_diffusion <- function(data, sigma = 1.0) {

    data_matrix <- as.matrix(data)
    data_matrix <- t(as.matrix(data))
    storage.mode(data_matrix) <- "double"
    .Call("c_run_diffusion", data_matrix, as.numeric(sigma), PACKAGE="cDiffusion")
}