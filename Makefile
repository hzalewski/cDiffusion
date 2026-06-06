.PHONY: all document install test check build clean site


all: document clean install test

document:
	Rscript -e "roxygen2::roxygenise('.')"

install:
	R CMD INSTALL .


test:
	Rscript -e "tinytest::test_package('cDiffusion')"

check: document clean
	R CMD build .
	R CMD check --as-cran --no-manual cDiffusion_*.tar.gz


site:
	Rscript -e "pkgdown::build_site()"

clean:
	rm -f src/*.o src/*.so
	rm -f cDiffusion_*.tar.gz
	rm -rf cDiffusion.Rcheck
