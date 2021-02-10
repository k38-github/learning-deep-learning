#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/function.h"
#include "../common/layers/AffineLayer.h"
#include "../common/layers/ReluLayer.h"
#include "../common/layers/SigmoidLayer.h"
#include "../common/layers/SoftmaxWithLossLayer.h"
#include "../common/layers/MultiLayerNet.h"
#include "../common/optimizer/SGD.h"
#include "../common/optimizer/Momentum.h"
#include "../common/optimizer/Nesterov.h"
#include "../common/optimizer/AdaGrad.h"
#include "../common/optimizer/RMSprop.h"
#include "../common/optimizer/Adam.h"
#include "../dataset/mnist.h"

int main(void) {

    MultiLayerNet multinet;

    char *X_TRAIN;
    char *T_TRAIN;
    char *X_TEST;
    char *T_TEST;
    int size[4] = {0};
    int one_hot_value = 10;

    load_mnist(&X_TRAIN, &T_TRAIN, &X_TEST, &T_TEST, size);

    double *x_train;
    x_train = (double *)malloc(sizeof(double) * size[0]);
    normalize(X_TRAIN, x_train, size[0]);

    int *t_train;
    t_train = (int *)malloc(sizeof(int) * size[1] * one_hot_value);
    one_hot(T_TRAIN, t_train, size[1]);

    double *x_test;
    x_test = (double *)malloc(sizeof(double) * size[2]);
    normalize(X_TEST, x_test, size[2]);

    int *t_test;
    t_test = (int *)malloc(sizeof(int) * size[3] * one_hot_value);
    one_hot(T_TEST, t_test, size[3]);

    int iters_num = 2000;
    int train_size = size[0];

    int input_size = 784;
    int hidden_size_list[4] = {100, 100, 100, 100};
    int hidden_layer_num = 4;
    // int hidden_size_list[2] = {100, 100};
    // int hidden_layer_num = 2;
    int output_size = 10;
    int batch_size = 128;

    char *activation = "relu";
    char *weight_init_std = "relu";
    double weight_decay_lambda = 0.0;

    multilayer_init(&multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda);

    int idx = 0;
    // SGD
    // double learning_rate = 0.01;
    // SGD optimizer;
    // sgd_init(&optimizer, learning_rate);

    // Momentum
    double learning_rate = 0.01;
    double momentum = 0.9;
    double **vW;
    double **vb;

    Momentum optimizer;
    momentum_init(&optimizer, learning_rate, momentum);

    vW = (double **)malloc(sizeof(double) * (multinet.hidden_layer_num + 1));
    vb = (double **)malloc(sizeof(double) * (multinet.hidden_layer_num + 1));

    for (idx=0;idx<multinet.hidden_layer_num+1;idx++) {
        vW[idx] = (double *)calloc(multinet.all_size_list[idx] * multinet.all_size_list[idx+1], sizeof(double));
        vb[idx] = (double *)calloc(multinet.all_size_list[idx+1], sizeof(double));
    }

    // Adagrad
    // double learning_rate = 1.5;
    // double **hW;
    // double **hb;

    // AdaGrad adagrad_optimizer;
    // adagrad_init(&adagrad_optimizer, learning_rate);

    // hW = (double **)malloc(sizeof(double) * (multinet.hidden_layer_num + 1));
    // hb = (double **)malloc(sizeof(double) * (multinet.hidden_layer_num + 1));

    // for (idx=0;idx<multinet.hidden_layer_num+1;idx++) {
    //     hW[idx] = (double *)calloc(multinet.all_size_list[idx] * multinet.all_size_list[idx+1], sizeof(double));
    //     hb[idx] = (double *)calloc(multinet.all_size_list[idx+1], sizeof(double));
    // }

    int *batch_mask;
    multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        printf("iters_num: %d\n", i);

        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        m = 0;
        k = 0;

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*input_size;k<(batch_mask[j]*input_size)+input_size;k++) {
                multinet.x_batch[l] = x_train[k];
                l++;
            }
        }

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {
                multinet.t_batch[m] = t_train[k];
                m++;
            }
        }

        gradient(&multinet, multinet.x_batch, multinet.t_batch);

        for (idx=0;idx<multinet.hidden_layer_num+1;idx++) {
            printf("idx:%d %d\n", multinet.all_size_list[idx], multinet.all_size_list[idx+1]);
            // SGD
            // sgd_update(&optimizer, multinet.W[idx], multinet.gW[idx], multinet.all_size_list[idx] * multinet.all_size_list[idx+1]);
            // sgd_update(&optimizer, multinet.b[idx], multinet.gb[idx], multinet.all_size_list[idx+1]);

            // Momentum
            momentum_update(&optimizer, multinet.W[idx], multinet.gW[idx], vW[idx], multinet.all_size_list[idx] * multinet.all_size_list[idx+1]);
            momentum_update(&optimizer, multinet.b[idx], multinet.gb[idx], vb[idx], multinet.all_size_list[idx+1]);

            // adagrad_update(&adagrad_optimizer, multinet.W[idx], multinet.gW[idx], hW[idx], multinet.all_size_list[idx] * multinet.all_size_list[idx+1]);
            // adagrad_update(&adagrad_optimizer, multinet.b[idx], multinet.gb[idx], hb[idx], multinet.all_size_list[idx+1]);

            memcpy(multinet.layers.Affine[idx].W, multinet.W[idx], sizeof(double) * multinet.all_size_list[idx] * multinet.all_size_list[idx+1]);
            memcpy(multinet.layers.Affine[idx].b, multinet.b[idx], sizeof(double) * multinet.all_size_list[idx+1]);
        }

    }

    return 0;
}
