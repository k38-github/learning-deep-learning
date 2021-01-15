#include <stdio.h>
#include <stdlib.h>
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

    double *y_tmp;
    y_tmp = (double *)malloc(sizeof(double) * this->row_size);

    double *t_tmp;
    t_tmp = (double *)malloc(sizeof(double) * this->row_size);

    double *ret_tmp;
    ret_tmp = (double *)malloc(sizeof(double) * this->col_size);

    for (i=0;i<this->col_size;i++) {
        for (j=0;j<this->row_size;j++) {
            y_tmp[j] = this->y[j+(this->row_size*i)];
            t_tmp[j] = t[j+(this->row_size*i)];
        }
        cross_entropy_error(y_tmp, t_tmp, loss[i], this->row_size);
    }


    return 0;
}

int softmaxwithlosslayer_backward(SoftmaxWithLossLayer *this, double *dx, double *dout) {

    double *W_trans;
    W_trans = (double *)malloc(sizeof(double) * this->w_row_size * this->w_col_size);

    // W_trans = W.T
    // (3, 2) <- (2, 3)
    trans_function(W_trans, this->W, this->w_col_size, this->w_row_size);

    // dx = dout x W_trans
    // (N, 2) = (N, 3) x (3, 2)
    dot_function(&dx, dout, W_trans, this->x_col_size, this->w_row_size, this->w_col_size);

    double *x_trans;
    x_trans = (double *)malloc(sizeof(double) * this->x_row_size * this->x_col_size);

    // x_trans = x.T
    // (2, N) <- (N, 2)
    trans_function(x_trans, this->x, this->x_col_size, this->x_row_size);

    // dW = x_trans x dout
    // (2, 3) = (2, N) x (N, 3)
    dot_function(&dx, dout, W_trans, this->x_col_size, this->w_row_size, this->w_col_size);

    double *x_row;
    x_row = (double *)malloc(sizeof(double) * this->x_col_size);

    double x_row_sum = 0.0;

    int i, j;
    for (i=0;i<this->x_row_size;i++) {
        for (j=0;j<this->x_col_size;j++) {
            x_row[j] = x_trans[j+(this->x_col_size*i)];
        }
        // x_row_sum = x_row[0] + x_row[1] + ... + x_row[x_col_size -1] +x_row[x_col_size]
        sum_function(x_row, &x_row_sum, this->x_col_size);
        // 1x3
        this->b[i] = x_row_sum;
        x_row_sum = 0.0;
    }

    return 0;
}
