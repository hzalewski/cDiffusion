#include "cDiffusion.h"

// Similarity matrix with kernel function
void apply_gauss(double *data, double *dist, int n, int dim, double sigma)
{

    double z = -1.0 / (2.0 * sigma * sigma); 

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        dist[i + i * n] = 1.0;

        for (int j = 0; j < i; j++)
        {
            double sum_sq = 0.0;

            for (int k = 0; k < dim; k++)
            {
                double diff = data[k + i * dim] - data[k + j * dim];
                sum_sq += diff * diff;
            }

            double w = exp(z * sum_sq);

            dist[i + j * n] = w;
            dist[j + i * n] = w;
        }
    }
}

// Calculates row sums, normalizes the kernel and then symmetrization 
void matrix_normalization(double *dist, double *D_sqrt, int n)
{
    double *q = malloc(n * sizeof(double));

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += dist[i + j * n];
        q[j] = sum;
    }

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            dist[i + j * n] /= (q[i] * q[j]);
        }
    }

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            sum += dist[i + j * n];
        D_sqrt[j] = sqrt(sum);
    }

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            dist[i + j * n] /= (D_sqrt[i] * D_sqrt[j]);
        }
    }

    free(q);
}

void random_matrix(double *X, int n, int m)
{

    GetRNGstate();

    for (int i = 0; i < n * m; i++)
    {
        X[i] = rnorm(0.0, 1.0);
    }

    PutRNGstate();
}

void matrix_multiplication(double *A, double *X, double *Y, int n, int m)
{

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < n; k++)
            {
                sum += A[k + i * n] * X[k + j * n];
            }

            Y[i + j * n] = sum;
        }
    }
}

void ortogonalize(double *X, int n, int m)
{

    for (int j = 0; j < m; j++)
    {

        double norm = 0.0;
        for (int i = 0; i < n; i++)
        {

            double val = X[i + j * n];

            norm += val * val;
        }
        norm = sqrt(norm);
        double inv_norm = 1/norm;

        for (int i = 0; i < n; i++)
        {
            X[i + j * n] *= inv_norm;
        }

#pragma omp parallel for
        for (int k = j + 1; k < m; k++)
        {

            double vec_prod = 0.0;
            for (int i = 0; i < n; i++)
            {
                vec_prod += X[i + j * n] * X[i + k * n];
            }

            for (int i = 0; i < n; i++)
            {
                X[i + k * n] -= vec_prod * X[i + j * n];
            }
        }
    }
}

// Find eigenvectors for small matrix then projects them onto original n dimensions
void find_eigenvectors(double *X, double *Y, double *eigenvectors, double *eigenvalues, int n, int m)
{

    double *B = malloc(m * m * sizeof(double));

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += X[k + i * n] * Y[k + j * n];
            }
            B[i + j * m] = sum;
        }
    }

    char jobz = 'V';
    char uplo = 'U';
    int lwork = 3 * m - 1;
    double *work = malloc(lwork * sizeof(double));
    int info;

    dsyev_(&jobz, &uplo, &m, B, &m, eigenvalues, work, &lwork, &info, 1, 1);

    if (info != 0)
    {
        free(B);
        free(work);
        Rf_error("LAPACK dsyev_ error: %d", info);
    }

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < m; k++)
            {
                sum += X[i + k * n] * B[k + j * m];
            }
            eigenvectors[i + j * n] = sum;
        }
    }

    free(B);
    free(work);
}

// Randimized Singular Value Decomposition with power iteration
void randomized_svd(double *A, double *X, double *eigenvectors, double *eigenvalues, int n, int m, int n_iter)
{

    double *Y = malloc(n * m * sizeof(double));
    double *Y_mem = Y;

    for (int iter = 0; iter < n_iter; iter++)
    {

        matrix_multiplication(A, X, Y, n, m);

        ortogonalize(Y, n, m);

        double *temp = X;
        X = Y;
        Y = temp;
    }

    matrix_multiplication(A, X, Y, n, m);

    find_eigenvectors(X, Y, eigenvectors, eigenvalues, n, m);

    free(Y_mem);
}