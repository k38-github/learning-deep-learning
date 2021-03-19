#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "MultiLayerNetExtend.h"
#include "../function.h"

int multilayerextend_init(MultiLayerNetExtend *this, int input_size, int *hidden_size_list, int hidden_layer_num, int output_size, int batch_size, char *activation, char *weight_init_std, double weight_decay_lambda, char *use_dropout, double dropout_ration, char *use_batchnorm) {

    int i, j;

    this->input_size = input_size;
    this->output_size = output_size;
    this->batch_size = batch_size;
    this->hidden_size_list = (int *)malloc(sizeof(int) * hidden_layer_num);
    for (i=0;i<hidden_layer_num;i++) {
        this->hidden_size_list[i] = hidden_size_list[i];
    }
    this->hidden_layer_num = hidden_layer_num;
    this->weight_decay_lambda = weight_decay_lambda;

    this->all_size_list = (int *)malloc(sizeof(int) * (this->hidden_layer_num + 2));
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

    multilayerextend_init_weight(this, weight_init_std);

    // Layers
    this->layers.Affine = (AffineLayer *)malloc(sizeof(AffineLayer) * (hidden_layer_num + 1));
    this->layers.BatchNormalization = (BatchNormalization *)malloc(sizeof(BatchNormalization) * hidden_layer_num);
    this->layers.Relu = (ReluLayer *)malloc(sizeof(ReluLayer) * hidden_layer_num);
    this->layers.Sigmoid = (SigmoidLayer *)malloc(sizeof(SigmoidLayer) * hidden_layer_num);
    this->layers.Dropout = (Dropout *)malloc(sizeof(Dropout) * hidden_layer_num);

    this->activation = activation;
    this->use_dropout = use_dropout;
    this->use_batchnorm = use_batchnorm;

    for (i=0;i<hidden_layer_num+1;i++) {
        affinelayer_init(&this->layers.Affine[i], this->W[i], this->b[i], this->all_size_list[i], this->all_size_list[i+1]);
        //printf("%d:%p\n", i, &this->layers.Affine[i]);

        if (i != hidden_layer_num) {
            if (strcmp(use_batchnorm, "true") == 0) {
                double *gamma;
                double *beta;
                double momentum = 0.9;
                double *running_mean;
                double *running_var;

                gamma = (double *)malloc(sizeof(double) * this->all_size_list[i+1]);
                for (j=0;j<this->all_size_list[i+1];j++) {
                    gamma[j] = 1;
                }
                beta = (double *)calloc(this->all_size_list[i+1], sizeof(double));
                running_mean = (double *)malloc(sizeof(double) * this->all_size_list[i+1]);
                running_var = (double *)malloc(sizeof(double) * this->all_size_list[i+1]);

                batchnormalization_init(&this->layers.BatchNormalization[i], gamma, beta, momentum, running_mean, running_var, batch_size, this->all_size_list[i+1]);
            }

            if (strcmp(activation, "relu") == 0) {
                relulayer_init(&this->layers.Relu[i], batch_size * this->all_size_list[i+1]);
            } else if (strcmp(activation, "sigmoid") == 0) {
                sigmoidlayer_init(&this->layers.Sigmoid[i], batch_size * this->all_size_list[i+1]);
            }

            if (strcmp(use_dropout, "true") == 0) {
                double dropout_ratio = 0.5;

                dropout_init(&this->layers.Dropout[i], dropout_ratio, batch_size, this->all_size_list[i+1]);
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

    this->ggamma = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    this->gbeta = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (i=0;i<this->hidden_layer_num+1;i++) {
        this->ggamma[i] = (double *)malloc(sizeof(double) * this->all_size_list[i+1]);
        this->gbeta[i] = (double *)malloc(sizeof(double) * this->all_size_list[i+1]);
    }

    return 0;
}

int multilayerextend_init_weight(MultiLayerNetExtend *this, char *weight_init_std) {
    int i, j;
    double scale = 0.01;

    for (i=0;i<this->hidden_layer_num+1;i++) {
        if (strcmp(weight_init_std, "relu") == 0 || strcmp(weight_init_std, "he") == 0) {
            scale = sqrt(2.0 / this->all_size_list[i]);
        } else if (strcmp(weight_init_std, "sigmoid") == 0 || strcmp(weight_init_std, "xavier") == 0) {
            scale = sqrt(1.0 / this->all_size_list[i]);
        }

        random_randn(this->W[i], this->all_size_list[i], this->all_size_list[i+1]);

        // printf("i:%d size:%d scale:%f\n", i, this->all_size_list[i], scale);
        for (j=0;j<this->all_size_list[i]*this->all_size_list[i+1];j++) {
            this->W[i][j] = scale * this->W[i][j];
        }
    }

    return 0;
}

int multilayerextend_free(MultiLayerNetExtend *this) {
    free(this->hidden_size_list);
    free(this->all_size_list);

    int i;
    for (i=0;i<this->hidden_layer_num+1;i++) {
        free(this->W[i]);
        free(this->b[i]);
    }
    free(this->W);
    free(this->b);

    for (i=0;i<this->hidden_layer_num+1;i++) {
        affinelayer_free(&this->layers.Affine[i]);

        if (i != this->hidden_layer_num) {
            if (strcmp(this->activation, "relu") == 0) {
                relulayer_free(&this->layers.Relu[i]);
            } else if (strcmp(this->activation, "sigmoid") == 0) {
                sigmoidlayer_free(&this->layers.Sigmoid[i]);
            }
        }
    }

    free(this->layers.Affine);
    free(this->layers.Relu);
    free(this->layers.Sigmoid);

    softmaxwithlosslayer_free(&this->layers.SoftmaxWithLoss);

    for (i=0;i<this->hidden_layer_num+1;i++) {
        free(this->gW[i]);
        free(this->gb[i]);
    }

    free(this->gW);
    free(this->gb);

    return 0;
}

int predict(MultiLayerNetExtend *this, double *y, double *x, char *train_flg) {

    int i = 0;

    double **affine_ret;
    double **batchnorm_ret;
    double **relu_ret;
    double **dropout_ret;

    affine_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    batchnorm_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    relu_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    dropout_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);

    affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    batchnorm_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    relu_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    dropout_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

    // printf("forward------\n");
    // printf("Affine[%d]: x->affine_ret[%d]\n", i, i);
    affinelayer_forward(&this->layers.Affine[i], affine_ret[i], x, this->batch_size, this->all_size_list[i]);
    // printf("Relu[%d]: affine_ret[%d]->reluret[%d]\n", i, i, i);
    if (strcmp(this->use_batchnorm, "true") == 0) {
        batchnormalization_forward(&this->layers.BatchNormalization[i], batchnorm_ret[i], affine_ret[i], train_flg);
        relulayer_forward(&this->layers.Relu[i], relu_ret[i], batchnorm_ret[i]);
    } else {
        relulayer_forward(&this->layers.Relu[i], relu_ret[i], affine_ret[i]);
    }
    // print_matrix(relu_ret[i], this->batch_size, this->all_size_list[i], "e");
    // printf("\n");


    for (i=1;i<this->hidden_layer_num;i++) {
        affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        batchnorm_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        relu_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        dropout_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

        // printf("Affine[%d]: relu_ret[%d]->affine_ret[%d]\n", i, i-1, i);
        if (strcmp(this->use_dropout, "true") == 0) {
            dropout_forward(&this->layers.Dropout[i-1], dropout_ret[i-1], relu_ret[i-1], train_flg);
            affinelayer_forward(&this->layers.Affine[i], affine_ret[i], dropout_ret[i-1], this->batch_size, this->all_size_list[i]);
        } else {
            affinelayer_forward(&this->layers.Affine[i], affine_ret[i], relu_ret[i-1], this->batch_size, this->all_size_list[i]);
        }
        // printf("Relu[%d]: affine_ret[%d]->reluret[%d]\n", i, i, i);
        if (strcmp(this->use_batchnorm, "true") == 0) {
            batchnormalization_forward(&this->layers.BatchNormalization[i], batchnorm_ret[i], affine_ret[i], train_flg);
            relulayer_forward(&this->layers.Relu[i], relu_ret[i], batchnorm_ret[i]);
        } else {
            relulayer_forward(&this->layers.Relu[i], relu_ret[i], affine_ret[i]);
        }
        // printf("%d:%p %d\n", i, &this->layers.Affine[i], this->all_size_list[i]);
    }
    // printf("%d:%p %d\n", i, &this->layers.Affine[i], this->all_size_list[i]);
    // printf("Affine[%d]: relu_ret[%d]->affine_ret[%d] size: %d\n", i, i-1, i, this->all_size_list[i]);
    if (strcmp(this->use_dropout, "true") == 0) {
        dropout_forward(&this->layers.Dropout[i-1], dropout_ret[i-1], relu_ret[i-1], train_flg);
        affinelayer_forward(&this->layers.Affine[i], y, dropout_ret[i-1], this->batch_size, this->all_size_list[i]);
    } else {
        affinelayer_forward(&this->layers.Affine[i], y, relu_ret[i-1], this->batch_size, this->all_size_list[i]);
    }

    for (i=0;i<this->hidden_layer_num;i++) {
        free(affine_ret[i]);
        free(batchnorm_ret[i]);
        free(relu_ret[i]);
        free(dropout_ret[i]);
    }

    free(affine_ret);
    free(batchnorm_ret);
    free(relu_ret);
    free(dropout_ret);

    return 0;
}

int loss(MultiLayerNetExtend *this, double *ret, double *x, double *t, char *train_flg) {

    double *y;
    y = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    predict(this, y, x, train_flg);

    softmaxwithlosslayer_forward(&this->layers.SoftmaxWithLoss, ret, y, t);
    // printf("softmax_forward: %f\n", *ret);

    int i, j;
    double weight_decay = 0.0;
    double sum_W = 0.0;
    double *tmp;
    double *tmp_W;

    for (i=0;i<this->hidden_layer_num+1;i++) {
        if (i == 0) {
            tmp_W = (double *)malloc(sizeof(double) * this->all_size_list[i] * this->all_size_list[i+1]);
        } else {

            if ((tmp = (double *)realloc(tmp_W, sizeof(double) * this->all_size_list[i] * this->all_size_list[i+1])) == NULL) {
                printf("Unable to allocate memory during realloc\n");
                exit(EXIT_FAILURE);
            } else {
                tmp_W = tmp;
            }

        }

        for (j=0;j<this->all_size_list[i]*this->all_size_list[i+1];j++) {
            tmp_W[j] = pow(this->W[i][j], 2.0);
        }
        sum_function(tmp_W, &sum_W, this->all_size_list[i]*this->all_size_list[i+1]);
        weight_decay += 0.5 * this->weight_decay_lambda * sum_W;
        // printf("weight_decay: %.18f\n", weight_decay);
    }

    *ret += weight_decay;

    free(y);
    free(tmp_W);

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
int gradient(MultiLayerNetExtend *this, double *x, double *t) {
    // forward
    double loss_ret = 0.0;
    char *train_flg = "true";

    loss(this, &loss_ret, x, t, train_flg);

    // backward
    int i, j;
    double *dout;
    double *softmaxwithloss_ret;
    double **affine_ret;
    double **dropout_ret;
    double **relu_ret;
    double **batchnorm_ret;

    dout = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);
    softmaxwithloss_ret = (double *)malloc(sizeof(double) * this->batch_size * this->output_size);

    for(i=0;i<this->batch_size*this->output_size;i++) {
        dout[i] = 1;
    }

    softmaxwithlosslayer_backward(&this->layers.SoftmaxWithLoss, softmaxwithloss_ret, dout);

    affine_ret = (double **)malloc(sizeof(double) * (this->hidden_layer_num + 1));
    dropout_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    relu_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);
    batchnorm_ret = (double **)malloc(sizeof(double) * this->hidden_layer_num);

    i = this->hidden_layer_num;
    affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    dropout_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    relu_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
    batchnorm_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);


    // printf("backward------\n");
    // printf("Affine[%d]: softmax_ret->affine_ret[%d]\n", i, i);
    affinelayer_backward(&this->layers.Affine[i], affine_ret[i], softmaxwithloss_ret);
    // printf("Relu[%d]: affine_ret[%d]->reluret[%d]\n", i-1, i, i-1);
    if (strcmp(this->use_dropout, "true") == 0) {
        dropout_backward(&this->layers.Dropout[i-1], dropout_ret[i-1], affine_ret[i]);
        relulayer_backward(&this->layers.Relu[i-1], relu_ret[i-1], dropout_ret[i-1]);
    } else {
        relulayer_backward(&this->layers.Relu[i-1], relu_ret[i-1], affine_ret[i]);
    }

    for (i=this->hidden_layer_num-1;0<i;i--) {
        affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        dropout_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        relu_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);
        batchnorm_ret[i-1] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

        if (strcmp(this->use_batchnorm, "true") == 0) {
            batchnormalization_backward(&this->layers.BatchNormalization[i], batchnorm_ret[i], relu_ret[i]);
            affinelayer_backward(&this->layers.Affine[i], affine_ret[i], batchnorm_ret[i]);
        } else {
            // printf("Affine[%d]: relu_ret[%d]->affine_ret[%d]\n", i, i, i);
            affinelayer_backward(&this->layers.Affine[i], affine_ret[i], relu_ret[i]);
        }

        if (strcmp(this->use_dropout, "true") == 0) {
            dropout_backward(&this->layers.Dropout[i-1], dropout_ret[i-1], affine_ret[i]);
            relulayer_backward(&this->layers.Relu[i-1], relu_ret[i-1], dropout_ret[i-1]);
        } else {
            // printf("Relu[%d]: affine_ret[%d]->reluret[%d]\n", i-1, i, i-1);
            relulayer_backward(&this->layers.Relu[i-1], relu_ret[i-1], affine_ret[i]);
        }
        //printf("i: %d\n", i);
    }
    //printf("i: %d\n", i);
    affine_ret[i] = (double *)malloc(sizeof(double) * this->batch_size * this->all_size_list[i]);

    if (strcmp(this->use_batchnorm, "true") == 0) {
        batchnormalization_backward(&this->layers.BatchNormalization[i], batchnorm_ret[i], relu_ret[i]);
        affinelayer_backward(&this->layers.Affine[i], affine_ret[i], batchnorm_ret[i]);
    } else {
        // printf("Affine[%d]: relu_ret[%d]->affine_ret[%d]\n", i, i, i);
        affinelayer_backward(&this->layers.Affine[i], affine_ret[i], relu_ret[i]);
    }

    for (i=0;i<this->hidden_layer_num+1;i++) {
        for (j=0;j<this->all_size_list[i]*this->all_size_list[i+1];j++) {
            this->layers.Affine[i].dW[j] = this->layers.Affine[i].dW[j] + this->weight_decay_lambda * this->layers.Affine[i].W[j];
        }
    }

    for (i=0;i<this->hidden_layer_num+1;i++) {
        memcpy(this->gW[i], this->layers.Affine[i].dW, sizeof(double) * this->all_size_list[i] * this->all_size_list[i+1]);
        memcpy(this->gb[i], this->layers.Affine[i].db, sizeof(double) * this->all_size_list[i+1]);
    }

    if (strcmp(this->use_batchnorm, "true") == 0) {
        for (i=0;i<this->hidden_layer_num+1;i++) {
            memcpy(this->ggamma[i], this->layers.BatchNormalization[i].dgamma, sizeof(double) * this->all_size_list[i+1]);
            memcpy(this->gbeta[i], this->layers.BatchNormalization[i].dbeta, sizeof(double) * this->all_size_list[i+1]);
        }
    }

    for (i=0;i<this->hidden_layer_num;i++) {
        free(affine_ret[i]);
        free(batchnorm_ret[i]);
        free(relu_ret[i]);
        free(dropout_ret[i]);
    }

    free(affine_ret);
    free(batchnorm_ret);
    free(relu_ret);
    free(dropout_ret);

    free(softmaxwithloss_ret);
    free(dout);

    return 0;
}
