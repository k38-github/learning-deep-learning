#include <stdio.h>
#include <stdlib.h>
#include "AffineLayer.h"
#include "../function.h"

int affinelayer_init(AffineLayer *this, int col_size, int row_size) {
    this->W = (double *)malloc(sizeof(double) * col_size * row_size);
    this->b = (double *)calloc(row_size, sizeof(double));

    this->dW = (double *)malloc(sizeof(double) * col_size * row_size);
    this->db = (double *)calloc(row_size, sizeof(double));

    this->w_col_size = col_size;
    this->w_row_size = row_size;

    return 0;
}

int affinelayer_forward(AffineLayer *this, double *out, double *x, int col_size, int row_size) {
    double *W_dot;
    W_dot = (double *)malloc(sizeof(double) * this->w_col_size * this->w_row_size);

    // W_dot = x x W
    // (N, 3) = (N, 2) x (2x3)
    dot_function(&W_dot, x, this->W, col_size, this->w_col_size, this->w_row_size);

    double *broadcast_b;
    broadcast_b = (double *)malloc(sizeof(double) * col_size * this->w_row_size);

    // broadcast_b = b
    // (N, 3) <- (1, 3)
    int i, j;
    for (i=0;i<col_size;i++) {
        for (j=0;j<this->w_row_size;j++) {
            broadcast_b[j+(i*this->w_row_size)] = this->b[j];
        }
    }

    // out = W_dot + b
    // (N, 3) = (N, 3) + (N, 3)
    matrix_sum(&out, W_dot, broadcast_b, col_size, this->w_row_size);

    this->x = (double *)malloc(sizeof(double) * col_size * row_size);
    this->x_col_size = col_size;
    this->x_row_size = row_size;

    for (i=0;i<col_size;i++) {
        for (j=0;j<row_size;j++) {
            this->x[j+(i*row_size)] = x[j];
        }
    }


    free(W_dot);
    free(broadcast_b);

    return 0;
}

int affinelayer_backward(AffineLayer *this, double *dx, double *dout) {

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
