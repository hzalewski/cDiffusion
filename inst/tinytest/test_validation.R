
valid_mat <- matrix(rnorm(100), 10, 10) 

expect_error(run_diffusion(data = "tekst", sigma = 1), pattern = "must be a numeric matrix")
expect_error(suppressWarnings(run_diffusion(data = matrix(1:3, nrow = 1), sigma = 1)), pattern = "too small")

bad_mat_na <- valid_mat
bad_mat_na[1, 1] <- NA
expect_error(run_diffusion(bad_mat_na, sigma = 1), pattern = "cannot contain NA")

bad_mat_inf <- valid_mat
bad_mat_inf[2, 2] <- Inf
expect_error(run_diffusion(bad_mat_inf, sigma = 1), pattern = "cannot contain Infinite")

expect_error(run_diffusion(valid_mat, sigma = -5, oversampling = 2), pattern = "sigma must be greater than zero")

expect_warning(run_diffusion(valid_mat, sigma = 1, k = 2, oversampling = 15), pattern = "Reducing oversampling")

expect_error(run_diffusion(valid_mat, sigma = 1, k = 1, oversampling = 2, n_iter = 0), pattern = "n_iter must be greater than zero")

