#include <stdio.h>
#include <stdlib.h>
#include "TwoLayerNet.h"
#include "../../common/function.h"

int init(TwoLayerNet *this, int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std) {

    this->W1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    random_randn(this->W1, input_size, hidden_size);

    this->b1 = (double *)calloc(hidden_size, sizeof(double));

    this->W2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    random_randn(this->W2, hidden_size, output_size);

    this->b2 = (double *)calloc(output_size, sizeof(double));

    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->weight_init_std = weight_init_std;

    return 0;
};


int predict(TwoLayerNet *this, double *y, double *x) {
    double *a1_dot;
    double *a1;
    a1_dot = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    a1 = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

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

    return 0;
};

int loss(TwoLayerNet *this, double *ret, double *x, double *t) {
    return 0;
}


