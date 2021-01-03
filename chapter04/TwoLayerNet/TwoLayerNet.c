#include <stdio.h>
#include <stdlib.h>
#include "TwoLayerNet.h"
#include "../../common/function.h"

int init(TwoLayerNet *this, int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std) {

    int i;

    // 784x100
    this->W1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    random_randn(this->W1, input_size, hidden_size);

    for (i=0;i<input_size*hidden_size;i++) {
        this->W1[i] = weight_init_std * this->W1[i];
    }

    // 100
    this->b1 = (double *)calloc(hidden_size, sizeof(double));

    // 100x10
    this->W2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    random_randn(this->W2, hidden_size, output_size);

    for (i=0;i<hidden_size*output_size;i++) {
        this->W2[i] = weight_init_std * this->W2[i];
    }

    // 10
    this->b2 = (double *)calloc(output_size, sizeof(double));

    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->weight_init_std = weight_init_std;

    this->gW1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    this->gb1 = (double *)malloc(sizeof(double) * hidden_size);
    this->gW2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    this->gb2 = (double *)malloc(sizeof(double) * output_size);

    return 0;
};


int predict(TwoLayerNet *this, double *y, double *x) {
    double *a1_dot;
    double *a1;
    a1_dot = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    a1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // a1_dot = x x W1
    // 100x100 = (100x784) x (784x100)
    dot_function(&a1_dot, x, this->W1, this->batch_size, this->input_size, this->hidden_size);
    matrix_sum(&a1, a1_dot, this->b1, this->batch_size, this->hidden_size);

    //print_matrix(a1, 100, 100, "f");
    //printf("\n");

    double *z1;
    z1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    sigmoid_function(a1, z1, this->batch_size * this->hidden_size);

    //print_matrix(z1, 100, 100, "f");
    //printf("\n");

    double *a2_dot;
    double *a2;
    a2_dot = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    a2 = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    // a2_dot = z1 x W2
    // 100x10 = (100x100) x (100x10)
    dot_function(&a2_dot, z1, this->W2, this->batch_size, this->hidden_size, this->output_size);
    matrix_sum(&a2, a2_dot, this->b2, this->batch_size, this->output_size);

    //print_matrix(a2, 100, 10, "f");
    //printf("\n");

    softmax_measures_function(a2, y, this->batch_size * this->output_size);

    //print_matrix(y, 100, 10, "f");
    //printf("\n");

    free(a1_dot);
    free(a1);
    free(a2_dot);
    free(a2);

    return 0;
};

int loss(TwoLayerNet *this, double *ret, double *w, int e) {

    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, this->x_batch);

    cross_entropy_error(y, this->t_batch, ret , this->output_size);
    printf("cross_entropy_error: %.20f\n", *ret);

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
    a1_dot = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    a1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    // a1_dot = x x W1
    // 100x100 = (100x784) x (784x100)
    dot_function(&a1_dot, x, this->W1, this->batch_size, this->input_size, this->hidden_size);
    matrix_sum(&a1, a1_dot, this->b1, this->batch_size, this->hidden_size);

    double *z1;
    z1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    sigmoid_function(a1, z1, this->batch_size * this->hidden_size);

    double *a2_dot;
    double *a2;
    a2_dot = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    a2 = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    dot_function(&a2_dot, z1, this->W2, this->batch_size, this->hidden_size, this->output_size);
    matrix_sum(&a2, a2_dot, this->b2, this->batch_size, this->output_size);

    softmax_measures_function(a2, y, this->batch_size * this->output_size);

    // backward
    double *dy;

    free(a1_dot);
    free(a1);
    free(a2_dot);
    free(a2);

    return 0;
}
