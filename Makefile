all:
	Rscript -e 'roxygen2::roxygenise(".")'
	R CMD INSTALL .
