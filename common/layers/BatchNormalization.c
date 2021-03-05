#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BatchNormalization.h"
#include "../function.h"

int batchnormalization_init(BatchNormalization *this, double *gamma, double *beta, double momentum, double running_mean, double running_var, int col_size, int row_size) {

    this->gamma = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->gamma, gamma, sizeof(double) * row_size);

    this->beta = (double *)malloc(sizeof(double) * row_size);
    memcpy(this->beta, beta, sizeof(double) * row_size);

    this->b = (double *)calloc(row_size, sizeof(double));

    this->momentum = momentum;

    this->running_mean = running_mean;
    this->running_var = running_var;

    return 0;
}

int batchnormalization_free(BatchNormalization *this) {
    return 0;
}

int batchnormalization_forward(BatchNormalization *this, double *out, double *x, char *train_flg) {
    return 0;
}

int batchnormalization_backward(BatchNormalization *this, double *dx, double *dout) {
    return 0;
}
