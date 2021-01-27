#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AffineLayer.h"
#include "../function.h"

int affinelayer_init(AffineLayer *this, double *W, double *b, int col_size, int row_size) {
    this->W = (double *)malloc(sizeof(double) * col_size * row_size);
    this->b = (double *)calloc(row_size, sizeof(double));

    memcpy(this->W, W, sizeof(double) * col_size * row_size);
    memcpy(this->b, b, sizeof(double) * row_size);

    this->dW = (double *)malloc(sizeof(double) * col_size * row_size);
    this->db = (double *)calloc(row_size, sizeof(double));

    this->w_col_size = col_size;
    this->w_row_size = row_size;

    this->x = (double *)malloc(sizeof(double) * col_size * row_size);

    return 0;
}

int affinelayer_free(AffineLayer *this) {

    free(this->W);
    free(this->b);
    free(this->dW);
    free(this->db);

    return 0;
}

int affinelayer_forward(AffineLayer *this, double *out, double *x, int col_size, int row_size) {
    double *W_dot;
    W_dot = (double *)malloc(sizeof(double) * col_size * this->w_row_size);

    // W_dot = x x W
    // (100, 50) = (100, 784) x (784, 50)
    // (100, 10) = (100, 50) x (50, 10)
    dot_function(&W_dot, x, this->W, col_size, this->w_col_size, this->w_row_size);

    double *broadcast_b;
    broadcast_b = (double *)malloc(sizeof(double) * col_size * this->w_row_size);

    // broadcast_b = b
    // (100, 50) <- (1, 50)
    // (100, 10) <- (1, 10)
    int i, j;
    for (i=0;i<col_size;i++) {
        for (j=0;j<this->w_row_size;j++) {
            broadcast_b[j+(i*this->w_row_size)] = this->b[j];
        }
    }

    // out = W_dot + b
    // (100, 50) = (100, 50) + (100, 50)
    // (100, 10) = (100, 10) + (100, 10)
    matrix_sum(&out, W_dot, broadcast_b, col_size, this->w_row_size);

    double *tmp;

    if ((tmp = (double *)realloc(this->x, sizeof(double) * col_size * row_size)) == NULL) {
        printf("Unable to allocate memory during realloc\n");
        exit(EXIT_FAILURE);
    } else {
        this->x = tmp;
    }
    this->x_col_size = col_size;
    this->x_row_size = row_size;

    memcpy(this->x, x, sizeof(double) * col_size * row_size);

    free(W_dot);
    free(broadcast_b);

    return 0;
}

int affinelayer_backward(AffineLayer *this, double *dx, double *dout) {

    double *W_trans;
    W_trans = (double *)malloc(sizeof(double) * this->w_row_size * this->w_col_size);

    // W_trans = W.T
    // (10, 50) <- (50, 10)
    // (50, 784) <- (784, 50)
    trans_function(W_trans, this->W, this->w_col_size, this->w_row_size);

    // dx = dout x W_trans
    // (100, 50) = (100, 10) x (10, 50)
    // (100, 784) = (100, 50) x (50, 784)
    dot_function(&dx, dout, W_trans, this->x_col_size, this->w_row_size, this->w_col_size);

    double *x_trans;
    x_trans = (double *)malloc(sizeof(double) * this->x_row_size * this->x_col_size);

    // x_trans = x.T
    // (50, 100) <- (100, 50)
    // (784, 100) <- (100, 784)
    trans_function(x_trans, this->x, this->x_col_size, this->x_row_size);

    // dW = x_trans x dout
    // (50, 10) = (50, 100) x (100, 10)
    // (784, 50) = (784, 100) x (100, 50)
    dot_function(&this->dW, x_trans, dout, this->x_row_size, this->x_col_size, this->w_row_size);

    double *dout_trans;
    dout_trans = (double *)malloc(sizeof(double) * this->w_row_size * this->x_col_size);

    // dout_trans
    // (10, 100) <- (100, 10)
    // (50, 100) <- (100, 50)
    trans_function(dout_trans, dout, this->x_col_size, this->w_row_size);

    double *dout_row;
    dout_row = (double *)malloc(sizeof(double) * this->x_col_size);

    double dout_row_sum = 0.0;

    int i, j;
    for (i=0;i<this->w_row_size;i++) {
        for (j=0;j<this->x_col_size;j++) {
            dout_row[j] = dout_trans[j+(this->x_col_size*i)];
        }
        // dout_row_sum = dout_row[0] + dout_row[1] + ... + dout_row[dout_col_size -1] +dout_row[dout_col_size]
        sum_function(dout_row, &dout_row_sum, this->x_col_size);
        // 1x10
        this->db[i] = dout_row_sum;
        dout_row_sum = 0.0;
    }

    free(W_trans);
    free(x_trans);
    free(dout_trans);
    free(dout_row);

    return 0;
}
