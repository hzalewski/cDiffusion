#' @useDynLib cDiffusion
"_PACKAGE"

#' @title
#' Run diffusion map
#'
#' @description
#' Main package function.
#'
#' @param data numeric matrix
#'
#' @return
#' returns a numeric matrix
#'
#' @export
run_diffusion <- function(data) {
    .Call("c_run_diffusion", data, PACKAGE="cDiffusion")
}