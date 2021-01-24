#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SoftmaxWithLossLayer.h"
#include "../function.h"

int softmaxwithlosslayer_init(SoftmaxWithLossLayer *this, int col_size, int row_size) {
    this->y = (double *)malloc(sizeof(double) * col_size * row_size);
    this->t = (double *)malloc(sizeof(double) * col_size * row_size);

    // col:batch_size row:output_size
    this->col_size = col_size;
    this->row_size = row_size;

    return 0;
}

int softmaxwithlosslayer_free(SoftmaxWithLossLayer *this) {

    free(this->y);
    free(this->t);

    return 0;
}

int softmaxwithlosslayer_forward(SoftmaxWithLossLayer *this, double *loss, double *x, double *t) {

    double *s;
    s = (double *)malloc(sizeof(double) * this->row_size);

    double *s_tmp;
    s_tmp = (double *)malloc(sizeof(double) * this->row_size);

    // y = s
    // (100x10) = (100x10)
    int i, j;
    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            s_tmp[j] = x[j+(this->row_size*i)];
        }
        softmax_measures_function(s_tmp, s, this->row_size);

        for (j=0;j<this->row_size;j++) {
            this->y[j+(this->row_size*i)] = s[j];
        }
    }

    //cross_entropy_error(this->y, t, loss, this->row_size);

    // y (100x10)
    // t (100x10)
    double *y_tmp;
    y_tmp = (double *)malloc(sizeof(double) * this->row_size);

    double *t_tmp;
    t_tmp = (double *)malloc(sizeof(double) * this->row_size);

    double *loss_tmp;
    loss_tmp = (double *)malloc(sizeof(double) * this->col_size);

    // y = s
    // (100x10) = (100x10)
    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            y_tmp[j] = this->y[j+(this->row_size*i)];
            t_tmp[j] = t[j+(this->row_size*i)];
        }
        cross_entropy_error(y_tmp, t_tmp, loss, this->row_size);
        loss_tmp[i] = *loss;
    }

    sum_function(loss_tmp, loss, this->col_size);
    *loss = *loss / this->col_size;

    memcpy(this->t, t, sizeof(double) * this->col_size * this->row_size);

    free(s);
    free(s_tmp);
    free(y_tmp);
    free(t_tmp);
    free(loss_tmp);

    return 0;
}

int softmaxwithlosslayer_backward(SoftmaxWithLossLayer *this, double *dx, double *dout) {

    double *dx_diff;
    dx_diff = (double *)malloc(sizeof(double) * this->col_size * this->row_size);

    // dx_diff = this->y - this->t
    // (100, 10) = (100, 10) - (100, 10)
    matrix_diff(&dx_diff, this->y, this->t, this->col_size, this->row_size);

    // dx = dx_diff / this->col_size
    int i;
    for (i=0;i<this->col_size*this->row_size;i++) {
        dx[i] = dx_diff[i] / this->col_size;
    }

    return 0;
}
