set.seed(29)
N <- 50
P <- 5
mat <- matrix(rnorm(N * P), nrow = N, ncol = P)
dims_val <- 3
over_val <- 2 

res_dense <- run_diffusion(mat, sigma = 1.0, dims = dims_val, oversampling = over_val, n_iter = 5)
res_sparse <- run_diffusion_sparse(mat, dims = dims_val, oversampling = over_val, n_iter = 5, k_neighbors = 10)

expect_true(inherits(res_dense, "diffmap"), info = "Output should have class 'diffmap'")
expect_true(is.list(res_dense), info = "Output should be a list")

expected_names <- c("coordinates", "eigenvalues", "sigma", "data_dim", "method")
expect_true(all(expected_names %in% names(res_dense)), info = "List doesnt have correct names")

expect_equal(length(res_dense$eigenvalues), dims_val, info = "Number of eigenvalues doesnt match k")
expect_equal(dim(res_dense$coordinates), c(N, dims_val), info = "Result matrix is not N x k")
expect_equal(dim(res_sparse$coordinates), c(N, dims_val))


expect_true(res_dense$eigenvalues[1] >= res_dense$eigenvalues[2], info = "Eigenvalues are not sorted")
expect_true(res_sparse$eigenvalues[1] >= res_sparse$eigenvalues[2])

set.seed(2979)
res_run1 <- run_diffusion(mat, sigma = 1.0, dims = dims_val, oversampling = over_val, n_iter = 5)
set.seed(2979)
res_run2 <- run_diffusion(mat, sigma = 1.0, dims = dims_val, oversampling = over_val, n_iter = 5)

expect_equal(res_run1$eigenvalues, res_run2$eigenvalues, info = "Values are not equal with the same seed")
expect_equal(res_run1$coordinates, res_run2$coordinates, info = "Coordinates are not equal with the same seed")

est_sig <- estimate_sigma(mat)
expect_true(is.numeric(est_sig) && est_sig > 0, info = "estimate_sigma didn't return positive number")

res_c <- run_diffusion(mat, sigma = est_sig, dims = dims_val, oversampling = 7, n_iter = 100)


D_sq <- as.matrix(dist(mat))^2 
W <- exp(-D_sq / (2 * est_sig^2))

q <- rowSums(W)
W_alpha <- W / outer(q, q)

D_new <- rowSums(W_alpha)
D_sqrt <- sqrt(D_new)
P_sym <- W_alpha / outer(D_sqrt, D_sqrt) 

r_eigen <- eigen(P_sym, symmetric = TRUE)
r_values <- r_eigen$values[2:(dims_val+1)] 
r_vectors <- r_eigen$vectors[, 2:(dims_val+1)]


for(i in 1:dims_val) {
  r_vectors[, i] <- (r_vectors[, i] / D_sqrt) * r_values[i]
}

expect_equal(res_c$eigenvalues, r_values, tolerance = 1e-4, 
             info = "Eigenvalues should closely match base R")

for (i in 1:dims_val) {
    vector_c <- res_c$coordinates[, i]
    vector_r <- r_vectors[, i]
    
    corr <- cor(vector_c, vector_r)
    
    expect_true(abs(corr) > 0.98, 
                info = sprintf("Vector %d should be highly correlated with base R. Correlation: %f", i, corr))
}


set.seed(67)
c1 <- matrix(rnorm(150, mean = 0, sd = 0.5), ncol = 3)
c2 <- matrix(rnorm(150, mean = 10, sd = 0.5), ncol = 3)
c3 <- matrix(rnorm(150, mean = 20, sd = 0.5), ncol = 3)
datakmeans <- rbind(c1, c2, c3)

dataDM <- run_diffusion(datakmeans, sigma = 1.0, dims = 6, oversampling = 5)
dm_small <- suppressWarnings(run_diffusion(datakmeans, dims = 1, oversampling = 10))
expect_error(cdiff_kmeans(dm_small, centers = "auto"), pattern = "Not enough eigenvalues", info = "diffmap has < 3 eigenvalues")

km4 <- cdiff_kmeans(dataDM, centers = 4, nstart = 5)

expect_true(inherits(km4, "kmeans"), info = "Output isn't of class 'kmeans'")
expect_equal(length(km4$size), 4, info = "Didn't exactly 4 clusters when requested")