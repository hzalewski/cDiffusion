#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>

SEXP c_run_diffusion(SEXP r_data)
{
    if (!Rf_isReal(r_data)) Rf_error("data must be numeric");
    
    Rprintf("everything works\n");
    
    return r_data;
}