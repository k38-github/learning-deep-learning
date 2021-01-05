#include <stdio.h>
#include <stdlib.h>
#include "TwoLayerNet.h"
#include "../../common/function.h"

int init(TwoLayerNet *this, int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std) {

    int i = 0;

    // 784x50
    this->W1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    random_randn(this->W1, input_size, hidden_size);

    for (i=0;i<input_size*hidden_size;i++) {
        this->W1[i] = weight_init_std * this->W1[i];
    }

    // 100x50
    this->b1 = (double *)calloc(batch_size * hidden_size, sizeof(double));

    // 50x10
    this->W2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    random_randn(this->W2, hidden_size, output_size);

    for (i=0;i<hidden_size*output_size;i++) {
        this->W2[i] = weight_init_std * this->W2[i];
    }

    // 100x10
    this->b2 = (double *)calloc(batch_size * output_size, sizeof(double));

    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->weight_init_std = weight_init_std;

    // 784x50
    this->gW1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    // 100x50
    this->gb1 = (double *)malloc(sizeof(double) * batch_size * hidden_size);
    // 50x10
    this->gW2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    // 100x10
    this->gb2 = (double *)malloc(sizeof(double) * batch_size * output_size);

    return 0;
};


int predict(TwoLayerNet *this, double *y, double *x) {
    double *a1_dot;
    double *a1;
    a1_dot = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    a1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // a1_dot = x x W1
    // (100x50) = (100x784) x (784x50)
    dot_function(&a1_dot, x, this->W1, this->batch_size, this->input_size, this->hidden_size);

    // a1 = a1_dot + b1
    // (100x50) = (100x50) + (100x50)
    matrix_sum(&a1, a1_dot, this->b1, this->batch_size, this->hidden_size);

    double *z1;
    z1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // z1 = a1
    // (100x50) = (100x50)
    sigmoid_function(a1, z1, this->batch_size * this->hidden_size);

    double *a2_dot;
    double *a2;
    a2_dot = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    a2 = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    // a2_dot = z1 x W2
    // (100x10) = (100x50) x (50x10)
    dot_function(&a2_dot, z1, this->W2, this->batch_size, this->hidden_size, this->output_size);

    // a2 = a2_bot + b2
    // (100x10) = (100x10) + (100x10)
    matrix_sum(&a2, a2_dot, this->b2, this->batch_size, this->output_size);

    double *s;
    s = (double *)malloc(sizeof(double) * this->output_size);

    double *s_tmp;
    s_tmp = (double *)malloc(sizeof(double) * this->output_size);

    // y = s
    // (100x10) = (100x10)
    int i = 0;
    int j = 0;
    for (i=0;i<this->batch_size;i++) {
        for (j=0;j<this->output_size;j++) {
            s_tmp[j] = a2[j+(this->output_size*i)];
        }
        softmax_measures_function(s_tmp, s, this->output_size);

        for (j=0;j<this->output_size;j++) {
            y[j+(this->output_size*i)] = s[j];
        }
    }

    free(a1_dot);
    free(a1);
    free(z1);
    free(a2_dot);
    free(a2);
    free(s);
    free(s_tmp);

    return 0;
};

int loss(TwoLayerNet *this, double *ret, double *w, int e) {

    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, this->x_batch);
    cross_entropy_error(y, this->t_batch, ret , this->output_size);

    free(y);

    return 0;
}

int accuracy(TwoLayerNet *this) {
    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, this->x_batch);

    int a = 0;
    int b = 0;
    double tmp[10] = {0};
    int *y_ret;
    int *t_ret;
    y_ret = (int *)malloc(sizeof(int) * this->batch_size);
    t_ret = (int *)malloc(sizeof(int) * this->batch_size);

    int i;
    for (i=0;i<this->batch_size*this->output_size;i++) {
        tmp[a] = y[i];
        a++;
        if (a%10 == 0) {
            argmax(tmp, &y_ret[b], 10);
            b++;
            a = 0;
        }
    }

    a = 0;
    b = 0;
    for (i=0;i<this->batch_size*this->output_size;i++) {
        tmp[a] = this->t_batch[i];
        a++;
        if (a%10 == 0) {
            argmax(tmp, &t_ret[b], 10);
            b++;
            a = 0;
        }
    }

    double collect_count = 0.0;

    for (i=0;i<this->batch_size;i++) {
        if (y_ret[i] == t_ret[i]) {
            collect_count++;
        }
    }

    printf("accuracy: %20.18f\n", collect_count/this->batch_size);

    return 0;
}

int gradient(TwoLayerNet *this, double *x, double *t) {
    // forward
    double *a1_dot;
    double *a1;
    a1_dot = (double *)calloc(this->batch_size * this->hidden_size, sizeof(double));
    a1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // a1_dot = x x W1
    // (100x50) = (100x784) x (784x50)
    dot_function(&a1_dot, x, this->W1, this->batch_size, this->input_size, this->hidden_size);

    // a1 = a1_dot + b1 :ok
    // (100x50) = (100x50) + (100x50)
    matrix_sum(&a1, a1_dot, this->b1, this->batch_size, this->hidden_size);

    double *z1;
    z1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // z1 = a1
    // (100x50) = (100x50)
    sigmoid_function(a1, z1, this->batch_size * this->hidden_size);

    double *a2_dot;
    double *a2_tmp;
    double *a2;
    a2_dot = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    a2_tmp = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    a2 = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    // a2_dot = z1 x W2
    // 100x10 = (100x50) x (50x10)
    dot_function(&a2_dot, z1, this->W2, this->batch_size, this->hidden_size, this->output_size);

    // a2_tmp = a2_dot + b2
    // (100x10) = (100x10) + (100x10)
    matrix_sum(&a2_tmp, a2_dot, this->b2, this->batch_size, this->output_size);

    double *s;
    s = (double *)malloc(sizeof(double) * this->output_size);

    double *s_tmp;
    s_tmp = (double *)malloc(sizeof(double) * this->output_size);

    // a2 = s
    // (100x10) = (100x10)
    int i = 0;
    int j = 0;
    for (i=0;i<this->batch_size;i++) {
        for (j=0;j<this->output_size;j++) {
            s_tmp[j] = a2_tmp[j+(this->output_size*i)];
        }

        softmax_measures_function(s_tmp, s, this->output_size);
        for (j=0;j<this->output_size;j++) {
            a2[j+(this->output_size*i)] = s[j];
        }
    }

    // backward
    double *dy_diff;
    double *dy;
    dy_diff = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    dy = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    // dy_diff = a2 - t
    // (100x10) = (100x10) - (100x10)
    matrix_diff(&dy_diff, a2, t, this->batch_size, this->output_size);

    // dy = dy_diff / batch_size
    // (100x10) = (100x10) / 100
    for (i=0;i<this->batch_size*this->output_size;i++) {
        dy[i] = dy_diff[i] / this->batch_size;
    }


    double *z1_trans;
    z1_trans = (double *)malloc(sizeof(double) * this->hidden_size * this->batch_size);

    // z1_trans = z1.T
    // (50x100) <- (100x50)
    trans_function(z1_trans, z1, this->batch_size, this->hidden_size);

    // gW2 = z1_trans x dy
    // (50x10) = (50x100) x (100x10)
    dot_function(&this->gW2, z1_trans, dy, this->batch_size, this->hidden_size, this->output_size);

    double *dy_trans;
    dy_trans = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    // dy_trans = dy.T
    // (10x100) <- (100x10)
    trans_function(dy_trans, dy, this->batch_size, this->output_size);

    double *dy_row;
    dy_row = (double *)malloc(sizeof(double) * this->batch_size);

    double *gb2_tmp;
    gb2_tmp = (double *)malloc(sizeof(double) * this->output_size);

    double dy_row_sum = 0.0;

    for (i=0;i<this->output_size;i++) {
        for (j=0;j<this->batch_size;j++) {
            dy_row[j] = dy_trans[j+(this->batch_size*i)];
        }
        // dy_row_sum = dy_row[0] + dy_row[1] + ... + dy_row[batch_size -1] +dy_row[batch_size]
        sum_function(dy_row, &dy_row_sum, this->batch_size);
        // 1x10
        gb2_tmp[i] = dy_row_sum;
        dy_row_sum = 0.0;
    }

    // gb2 = gb2_tmp
    // (100x10) <- (1x10)
    for (i=0;i<this->batch_size;i++) {
        for (j=0;j<this->output_size;j++) {
            this->gb2[j+(i*this->output_size)] = gb2_tmp[j];
        }
    }

    double *W2_trans;
    W2_trans = (double *)malloc(sizeof(double) * this->output_size * this->hidden_size);

    // W2_trans = W2.T
    // (10x50) <- (50x10)
    trans_function(W2_trans, this->W2, this->hidden_size, this->output_size);

    double *dz1;
    dz1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // dz1 = dy x W2_trans
    // (100x50) = (100x10) x (10x50)
    dot_function(&dz1, dy, W2_trans, this->batch_size, this->output_size, this->hidden_size);

    double *da1_sigmoid;
    double *da1;
    da1_sigmoid = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    da1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // da1_sigmoid = a1
    // (100x50) <- (100x50)
    sigmoid_grad_function(a1, da1_sigmoid, this->batch_size * this->hidden_size);

    // da1 = da1_sigmoid * dz1
    // (100x50) = (100x50) * (100x50)
    for (i=0;i<this->batch_size*this->hidden_size;i++) {
        da1[i] = da1_sigmoid[i] * dz1[i];
    }

    double *x_trans;
    x_trans = (double *)malloc(sizeof(double) * this->input_size * this->batch_size);

    // x_trans = x.T
    // (784x100) <-(100x784)
    trans_function(x_trans, x, this->batch_size, this->input_size);

    // gW1 = x_trans x da1
    // (784x50) = (784x100) x (100x50)
    dot_function(&this->gW1, x_trans, da1, this->input_size, this->batch_size, this->hidden_size);

    double *da1_trans;
    da1_trans = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // da1_trans = da1.T
    // (50x100) <- (100x50)
    trans_function(da1_trans, da1, this->batch_size, this->hidden_size);

    double *da1_row;
    da1_row = (double *)malloc(sizeof(double) * this->batch_size);

    double *gb1_tmp;
    gb1_tmp = (double *)malloc(sizeof(double) * this->hidden_size);

    double da1_row_sum = 0.0;
    for (i=0;i<this->hidden_size;i++) {
        for (j=0;j<this->batch_size;j++) {
            da1_row[j] = da1_trans[j+(this->batch_size*i)];
        }

        // da1_row_sum = da1_row[0] + da1_row[1] + ... + da1_row[batch_size -1] +da1_row[batch_size]
        sum_function(da1_row, &da1_row_sum, this->batch_size);
        // 1x50
        gb1_tmp[i] = da1_row_sum;
        da1_row_sum = 0.0;
    }

    // gb1 = gb1_tmp
    // (100x50) <- (1x50)
    for (i=0;i<this->batch_size;i++) {
        for (j=0;j<this->hidden_size;j++) {
            this->gb1[j+(i*this->hidden_size)] = gb1_tmp[j];
        }
    }

    free(a1_dot);
    free(a1);
    free(a2_dot);
    free(a2_tmp);
    free(a2);
    free(s);
    free(s_tmp);
    free(dy_diff);
    free(dy);
    free(z1_trans);
    free(dy_trans);
    free(dy_row);
    free(gb2_tmp);
    free(W2_trans);
    free(dz1);
    free(da1_sigmoid);
    free(da1);
    free(x_trans);
    free(da1_trans);
    free(da1_row);
    free(gb1_tmp);

    return 0;
}
