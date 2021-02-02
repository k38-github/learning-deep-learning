#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "MultiLayerNet.h"
#include "../function.h"

int multilayer_init(MultiLayerNet *this, int input_size, int *hidden_size_list, int hidden_layer_num, int output_size, int batch_size, char *activation, char *weight_init_std, double weight_decay_lambda) {

    int i;

    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->hidden_size_list = (int *)malloc(sizeof(int) * hidden_layer_num);
    for (i=0;i<hidden_layer_num;i++) {
        this->hidden_size_list[i] = hidden_size_list[i];
    }
    this->hidden_layer_num = hidden_layer_num;
    this->weight_decay_lambda = weight_decay_lambda;

    this->all_size_list = (int *)malloc(sizeof(int) * this->hidden_layer_num+2);
    this->all_size_list[0] = input_size;

    for (i=0;i<this->hidden_layer_num;i++) {
        this->all_size_list[i+1] = hidden_size_list[i];
    }
    this->all_size_list[hidden_layer_num+1] = output_size;

    // all_layer_num = input + hidden_layer_num + output - 1
    this->W = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    this->b = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (i=0;i<this->hidden_layer_num+1;i++) {
        this->W[i] = (double *)malloc(sizeof(double) * this->all_size_list[i] * this->all_size_list[i+1]);
        this->b[i] = (double *)calloc(this->all_size_list[i+1], sizeof(double));
    }

    multilayer_init_weight(this, weight_init_std);

    // Layers
    this->layers.Affine = (AffineLayer *)malloc(sizeof(AffineLayer) * (hidden_layer_num + 1));
    this->layers.Relu = (ReluLayer *)malloc(sizeof(ReluLayer) * hidden_layer_num);
    this->layers.Sigmoid = (SigmoidLayer *)malloc(sizeof(SigmoidLayer) * hidden_layer_num);

    for (i=0;i<hidden_layer_num+1;i++) {
        affinelayer_init(&this->layers.Affine[i], this->W[i], this->b[i], this->all_size_list[i], this->all_size_list[i+1]);
        //printf("%d:%p\n", i, &this->layers.Affine[i]);

        if (i != hidden_layer_num) {
            if (strcmp(activation, "relu") == 0) {
                relulayer_init(&this->layers.Relu[i], batch_size * this->all_size_list[i+1]);
            } else if (strcmp(activation, "sigmoid") == 0) {
                sigmoidlayer_init(&this->layers.Sigmoid[i], batch_size * this->all_size_list[i+1]);
            }
        }
    }

    softmaxwithlosslayer_init(&this->layers.SoftmaxWithLoss, batch_size, output_size);

    this->gW = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    this->gb = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (i=0;i<this->hidden_layer_num+1;i++) {
        this->gW[i] = (double *)malloc(sizeof(double) * this->all_size_list[i] * this->all_size_list[i+1]);
        this->gb[i] = (double *)calloc(this->all_size_list[i+1], sizeof(double));
    }

    return 0;
}

int multilayer_init_weight(MultiLayerNet *this, char *weight_init_std) {
    int i, j;
    double scale = 0.01;

    for (i=0;i<this->hidden_layer_num+1;i++) {
        if (strcmp(weight_init_std, "relu") == 0 || strcmp(weight_init_std, "he") == 0) {
            scale = sqrt(2.0 / this->all_size_list[i]);
        } else if (strcmp(weight_init_std, "sigmoid") == 0 || strcmp(weight_init_std, "xavier") == 0) {
            scale = sqrt(1.0 / this->all_size_list[i]);
        }

        random_randn(this->W[i], this->all_size_list[i], this->all_size_list[i+1]);

        for (j=0;j<this->all_size_list[i]*this->all_size_list[i+1];j++) {
            this->W[i][j] = scale * this->W[i][j];
        }
    }

    return 0;
}

//int multilayer_free(MultiLayerNet *this) {
//    affinelayer_free(&this->layers.Affine1);
//    relulayer_free(&this->layers.Relu1);
//    affinelayer_free(&this->layers.Affine2);
//    softmaxwithlosslayer_free(&this->layers.SoftmaxWithLoss);
//
//    return 0;
//
//}

int predict(MultiLayerNet *this, double *y, double *x) {

    int i = 0;

    double **affine_ret;
    double **relu_ret;

    affine_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    relu_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);

    affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    relu_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

    affinelayer_forward(&this->layers.Affine[i], affine_ret[i], x, this->all_size_list[i], this->all_size_list[i+1]);
    relulayer_forward(&this->layers.Relu[i], relu_ret[i], affine_ret[i]);
    for (i=1;i<this->hidden_layer_num;i++) {
        affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        relu_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

        affinelayer_forward(&this->layers.Affine[i], affine_ret[i], relu_ret[i-1], this->all_size_list[i], this->all_size_list[i+1]);
        relulayer_forward(&this->layers.Relu[i], relu_ret[i], affine_ret[i]);
        //printf("%d:%p\n", i, &this->layers.Affine[i]);

    }
    //printf("%d:%p\n", i, &this->layers.Affine[i]);
    affinelayer_forward(&this->layers.Affine[i], y, relu_ret[i-1], this->all_size_list[i], this->all_size_list[i+1]);

    for (i=0;i<this->hidden_layer_num;i++) {
        free(affine_ret[i]);
        free(relu_ret[i]);
    }

    free(affine_ret);
    free(relu_ret);

    return 0;
}

int loss(MultiLayerNet *this, double *ret, double *x, double *t) {

    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, x);

    softmaxwithlosslayer_forward(&this->layers.SoftmaxWithLoss, ret, y, t);
    printf("cross_entropy: %.18f\n", *ret);

    free(y);

    return 0;
}

//int accuracy(TwoLayerNet *this, double *ret, double *x, int *t) {
//    double *y;
//    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
//
//    // x_train (60000, 784)
//    // x_test  (10000, 784)
//    predict(this, y, x);
//
//    int *y_tmp;
//    y_tmp = (int *)malloc(sizeof(int) * this->batch_size);
//
//    int *t_tmp;
//    t_tmp = (int *)malloc(sizeof(int) * this->batch_size);
//
//    double *arr_tmp;
//    arr_tmp = (double *)malloc(sizeof(double) * this->output_size);
//
//    int i, j;
//    for (i=0;i<this->batch_size;i++) {
//        for (j=0;j<this->output_size;j++) {
//            arr_tmp[j] = y[j+(this->output_size*i)];
//        }
//        argmax(arr_tmp, &y_tmp[i], this->output_size);
//
//        for (j=0;j<this->output_size;j++) {
//            arr_tmp[j] = t[j+(this->output_size*i)];
//        }
//        argmax(arr_tmp, &t_tmp[i], this->output_size);
//    }
//
//    int collect_num = 0;
//    for (i=0;i<this->batch_size;i++) {
//        if ((int)y_tmp[i] == (int)t_tmp[i]) {
//            collect_num++;
//        }
//    }
//
//    *ret = (double)collect_num / this->batch_size;
//
//    free(y);
//    free(y_tmp);
//    free(t_tmp);
//    free(arr_tmp);
//
//    return 0;
//}
//
int gradient(MultiLayerNet *this, double *x, double *t) {
    // forward
    double loss_ret = 0.0;

    loss(this, &loss_ret, x, t);
//
//    // backward
//    double *dout;
//    double *softmaxwithloss_ret;
//    double *affine2_ret;
//    double *relu1_ret;
//    double *affine1_ret;
//
//    dout = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
//    softmaxwithloss_ret = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
//    affine2_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
//    relu1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->hidden_size);
//    affine1_ret = (double *)malloc(sizeof(double) * this->batch_size * this->input_size);
//
//    int i;
//    for(i=0;i<this->batch_size*this->output_size;i++) {
//        dout[i] = 1;
//    }
//
//    softmaxwithlosslayer_backward(&this->layers.SoftmaxWithLoss, softmaxwithloss_ret, dout);
//    affinelayer_backward(&this->layers.Affine2, affine2_ret, softmaxwithloss_ret);
//    relulayer_backward(&this->layers.Relu1, relu1_ret, affine2_ret);
//    affinelayer_backward(&this->layers.Affine1, affine1_ret, relu1_ret);
//
//    memcpy(this->gW1, this->layers.Affine1.dW, sizeof(double) * this->input_size * this->hidden_size);
//    memcpy(this->gb1, this->layers.Affine1.db, sizeof(double) * this->hidden_size);
//    memcpy(this->gW2, this->layers.Affine2.dW, sizeof(double) * this->hidden_size * this->output_size);
//    memcpy(this->gb2, this->layers.Affine2.db, sizeof(double) * this->output_size);
//
//    free(dout);
//    free(softmaxwithloss_ret);
//    free(affine2_ret);
//    free(relu1_ret);
//    free(affine1_ret);
//
    return 0;
}
