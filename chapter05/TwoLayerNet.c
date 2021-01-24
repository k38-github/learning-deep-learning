#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "TwoLayerNet.h"
#include "../common/function.h"


int init(TwoLayerNet *this, int input_size, int hidden_size, int output_size, int batch_size, double weight_init_std) {

    int i = 0;

    // 784x50
    this->W1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    random_randn(this->W1, input_size, hidden_size);

    for (i=0;i<input_size*hidden_size;i++) {
        this->W1[i] = weight_init_std * this->W1[i];
    }

    // 1x50
    this->b1 = (double *)calloc(hidden_size, sizeof(double));

    // 50x10
    this->W2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    random_randn(this->W2, hidden_size, output_size);

    for (i=0;i<hidden_size*output_size;i++) {
        this->W2[i] = weight_init_std * this->W2[i];
    }

    // 1x10
    this->b2 = (double *)calloc(output_size, sizeof(double));

    // Layers
    affinelayer_init(&this->layers.Affine1, this->W1, this->b1, input_size, hidden_size);
    relulayer_init(&this->layers.Relu1, batch_size * hidden_size);
    affinelayer_init(&this->layers.Affine2, this->W2, this->b2, hidden_size, output_size);
    softmaxwithlosslayer_init(&this->layers.SoftmaxWithLoss, batch_size, output_size);

    // 784x50
    this->gW1 = (double *)malloc(sizeof(double) * input_size * hidden_size);
    // 1x50
    this->gb1 = (double *)malloc(sizeof(double) * hidden_size);
    // 50x10
    this->gW2 = (double *)malloc(sizeof(double) * hidden_size * output_size);
    // 1x10
    this->gb2 = (double *)malloc(sizeof(double) * output_size);

    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->weight_init_std = weight_init_std;

    return 0;
}

int layers_free(TwoLayerNet *this) {
    affinelayer_free(&this->layers.Affine1);
    relulayer_free(&this->layers.Relu1);
    affinelayer_free(&this->layers.Affine2);
    softmaxwithlosslayer_free(&this->layers.SoftmaxWithLoss);

    return 0;

}

int predict(TwoLayerNet *this, double *y, double *x) {

    double *affine1_ret;
    double *relu1_ret;

    affine1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    relu1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);

    affinelayer_forward(&this->layers.Affine1, affine1_ret, x, this->batch_size, this->input_size);
    relulayer_forward(&this->layers.Relu1, relu1_ret, affine1_ret);
    affinelayer_forward(&this->layers.Affine2, y, relu1_ret, this->batch_size, this->hidden_size);

    free(affine1_ret);
    free(relu1_ret);

    return 0;
}

int loss(TwoLayerNet *this, double *ret, double *x, double *t) {

    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, x);

    softmaxwithlosslayer_forward(&this->layers.SoftmaxWithLoss, ret, y, t);

    free(y);

    return 0;
}

int accuracy(TwoLayerNet *this, double *ret, double *x, int *t) {
    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, x);

    int *y_tmp;
    y_tmp = (int *)malloc(sizeof(int) * this->batch_size);

    int *t_tmp;
    t_tmp = (int *)malloc(sizeof(int) * this->batch_size);

    double *arr_tmp;
    arr_tmp = (double *)malloc(sizeof(double) * this->output_size);

    int i, j;
    for (i=0;i<this->batch_size;i++) {
        for (j=0;j<this->output_size;j++) {
            arr_tmp[j] = y[j+(this->output_size*i)];
        }
        argmax(arr_tmp, &y_tmp[i], this->output_size);

        for (j=0;j<this->output_size;j++) {
            arr_tmp[j] = t[j+(this->output_size*i)];
        }
        argmax(arr_tmp, &t_tmp[i], this->output_size);
    }

    int collect_num = 0;
    for (i=0;i<this->batch_size;i++) {
        if ((int)y_tmp[i] == (int)t_tmp[i]) {
            collect_num++;
        }
    }

    *ret = (double)collect_num / this->batch_size;

    free(y);
    free(y_tmp);
    free(t_tmp);
    free(arr_tmp);

    return 0;
}

int gradient(TwoLayerNet *this, double *x, double *t) {
    // forward
    double loss_ret = 0.0;

    loss(this, &loss_ret, x, t);

    // backward
    double *dout;
    double *softmaxwithloss_ret;
    double *affine2_ret;
    double *relu1_ret;
    double *affine1_ret;

    dout = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    softmaxwithloss_ret = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    affine2_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    relu1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
    affine1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->input_size);

    int i;
    for(i=0;i<this->batch_size*this->output_size;i++) {
        dout[i] = 1;
    }

    softmaxwithlosslayer_backward(&this->layers.SoftmaxWithLoss, softmaxwithloss_ret, dout);
    affinelayer_backward(&this->layers.Affine2, affine2_ret, softmaxwithloss_ret);
    relulayer_backward(&this->layers.Relu1, relu1_ret, affine2_ret);
    affinelayer_backward(&this->layers.Affine1, affine1_ret, relu1_ret);

    memcpy(this->gW1, this->layers.Affine1.dW, sizeof(double) * this->input_size * this->hidden_size);
    memcpy(this->gb1, this->layers.Affine1.db, sizeof(double) * this->hidden_size);
    memcpy(this->gW2, this->layers.Affine2.dW, sizeof(double) * this->hidden_size * this->output_size);
    memcpy(this->gb2, this->layers.Affine2.db, sizeof(double) * this->output_size);

    free(dout);
    free(softmaxwithloss_ret);
    free(affine2_ret);
    free(relu1_ret);
    free(affine1_ret);

    return 0;
}
