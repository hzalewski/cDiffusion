#include "diffusion_core.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif
#include <Rconfig.h>
#include <R_ext/Lapack.h>

void apply_gauss(double *data, double *dist, int n, int p, double sigma)
{
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        dist[i + i * n] = 1.0;

        for (int j = 0; j < i; j++)
        {
            double sum_sq = 0.0;

            for (int k = 0; k < p; k++)
            {
                double diff = data[k + i * p] - data[k + j * p];
                sum_sq += diff * diff;
            }

            double d = -1.0 * sum_sq / (2.0 * sigma * sigma);
            double w = exp(d);

            dist[i + j * n] = w;
            dist[j + i * n] = w;
        }
    }
}

void matrix_normalization(double *dist, int n)
{

    double *sums = (double *)malloc(n * sizeof(double));

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        double column_sums = 0.0;

        for (int i = 0; i < n; i++)
        {
            column_sums += dist[i + j * n];
        }
        sums[j] = column_sums;
    }

#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            dist[i + j * n] = dist[i + j * n] / (sqrt(sums[i]) * sqrt(sums[j]));
        }
    }

    free(sums);
}

int eigen(double *matrix, int n, double *eigenvalues, double *eigenvectors)
{
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int lwork = 3 * n - 1;
    int info = 0;

    double *work = (double *)malloc(lwork * sizeof(double));

    F77_CALL(dsyev)(&jobz, &uplo, &n, matrix, &lda, eigenvalues, work, &lwork, &info 1, 1);

    for (int i = 0; i < n * n; i++)
    {
        eigenvectors[i] = matrix[i];
    }

    free(work);
    return info;
}