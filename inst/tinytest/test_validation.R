
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

expect_warning(run_diffusion(valid_mat, sigma = 1, dims = 2, oversampling = 15), pattern = "Reducing oversampling")

expect_error(run_diffusion(valid_mat, sigma = 1, dims = 1, oversampling = 2, n_iter = 0), pattern = "n_iter must be greater than zero")

set.seed(67)
c1 <- matrix(rnorm(150, mean = 0, sd = 0.5), ncol = 3)
c2 <- matrix(rnorm(150, mean = 10, sd = 0.5), ncol = 3)
c3 <- matrix(rnorm(150, mean = 20, sd = 0.5), ncol = 3)
datakmeans <- rbind(c1, c2, c3)

dm_clustered <- run_diffusion(datakmeans, sigma = 1.0, dims = 6, oversampling = 5)

expect_error(cdiff_kmeans(datakmeans), pattern = "must be an object of class 'diffmap'", info = "input is not a diffmap object")

expect_error(cdiff_kmeans(dm_clustered, centers = "aaaaaalegiamistrz"), pattern = "must be 'auto' or a single integer", info = "invalid string for centers")

expect_error(cdiff_kmeans(dm_clustered, centers = 1), pattern = ">= 2", info = "k is less than 2")

uniform_mat <- matrix(rnorm(300), ncol = 3)
dm_uniform <- run_diffusion(uniform_mat, sigma = 5.0, dims = 6, oversampling = 5)


expect_message(km_uniform <- cdiff_kmeans(dm_uniform, centers = "auto", nstart = 5),
  pattern = "Eigengap heuristic: selected k =",
  info = "didnt output the standard Eigengap message"
)

expect_true(length(km_uniform$size) >= 2, info = "Must always find at least 2 clusters")
expect_true(inherits(km_uniform, "kmeans"), info = "Output isnt a valid kmeans object")