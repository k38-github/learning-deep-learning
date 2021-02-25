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

    MultiLayerNet sgd_multinet;
    MultiLayerNet momentum_multinet;
    MultiLayerNet adagrad_multinet;
    MultiLayerNet adam_multinet;

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
    double *sgd_train_loss;
    double *momentum_train_loss;
    double *adagrad_train_loss;
    double *adam_train_loss;
    double *iters_num_arr;

    sgd_train_loss = (double *)malloc(sizeof(double) * iters_num);
    momentum_train_loss = (double *)malloc(sizeof(double) * iters_num);
    adagrad_train_loss = (double *)malloc(sizeof(double) * iters_num);
    adam_train_loss = (double *)malloc(sizeof(double) * iters_num);
    iters_num_arr = (double *)malloc(sizeof(double) * iters_num);

    int input_size = 784;
    int hidden_size_list[4] = {100, 100, 100, 100};
    int hidden_layer_num = 4;
    int output_size = 10;
    int batch_size = 128;

    char *activation = "relu";
    char *weight_init_std = "relu";
    double weight_decay_lambda = 0.0;

    multilayer_init(&sgd_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda);
    multilayer_init(&momentum_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda);
    multilayer_init(&adagrad_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda);
    multilayer_init(&adam_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda);

    int idx = 0;
    // SGD
    double learning_rate = 0.01;
    SGD sgd_optimizer;
    sgd_init(&sgd_optimizer, learning_rate);

    // Momentum
    learning_rate = 0.01;
    double momentum = 0.9;
    double **vW;
    double **vb;

    Momentum momentum_optimizer;
    momentum_init(&momentum_optimizer, learning_rate, momentum);

    vW = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    vb = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (idx=0;idx<hidden_layer_num+1;idx++) {
        vW[idx] = (double *)calloc(momentum_multinet.all_size_list[idx] * momentum_multinet.all_size_list[idx+1], sizeof(double));
        vb[idx] = (double *)calloc(momentum_multinet.all_size_list[idx+1], sizeof(double));
    }

    // Adagrad
    learning_rate = 0.01;
    double **hW;
    double **hb;

    AdaGrad adagrad_optimizer;
    adagrad_init(&adagrad_optimizer, learning_rate);

    hW = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    hb = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (idx=0;idx<hidden_layer_num+1;idx++) {
        hW[idx] = (double *)calloc(adagrad_multinet.all_size_list[idx] * adagrad_multinet.all_size_list[idx+1], sizeof(double));
        hb[idx] = (double *)calloc(adagrad_multinet.all_size_list[idx+1], sizeof(double));
    }

    // Adam
    learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double **mW;
    double **mb;
    double **vvW;
    double **vvb;

    Adam adam_optimizer;
    adam_init(&adam_optimizer, learning_rate, beta1, beta2);

    mW = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    mb = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (idx=0;idx<hidden_layer_num+1;idx++) {
        mW[idx] = (double *)calloc(adam_multinet.all_size_list[idx] * adam_multinet.all_size_list[idx+1], sizeof(double));
        mb[idx] = (double *)calloc(adam_multinet.all_size_list[idx+1], sizeof(double));
    }

    vvW = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));
    vvb = (double **)malloc(sizeof(double) * (hidden_layer_num + 1));

    for (idx=0;idx<hidden_layer_num+1;idx++) {
        vvW[idx] = (double *)calloc(adam_multinet.all_size_list[idx] * adam_multinet.all_size_list[idx+1], sizeof(double));
        vvb[idx] = (double *)calloc(adam_multinet.all_size_list[idx+1], sizeof(double));
    }

    int *batch_mask;
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    sgd_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    sgd_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    momentum_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    momentum_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    adagrad_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    adagrad_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    adam_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    adam_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);


    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        // printf("iters_num: %d\n", i);

        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        m = 0;
        k = 0;

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*input_size;k<(batch_mask[j]*input_size)+input_size;k++) {
                sgd_multinet.x_batch[l] = x_train[k];
                momentum_multinet.x_batch[l] = x_train[k];
                adagrad_multinet.x_batch[l] = x_train[k];
                adam_multinet.x_batch[l] = x_train[k];
                l++;
            }
        }

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {
                sgd_multinet.t_batch[m] = t_train[k];
                momentum_multinet.t_batch[m] = t_train[k];
                adagrad_multinet.t_batch[m] = t_train[k];
                adam_multinet.t_batch[m] = t_train[k];
                m++;
            }
        }

        gradient(&sgd_multinet, sgd_multinet.x_batch, sgd_multinet.t_batch);
        gradient(&momentum_multinet, momentum_multinet.x_batch, momentum_multinet.t_batch);
        gradient(&adagrad_multinet, adagrad_multinet.x_batch, adagrad_multinet.t_batch);
        gradient(&adam_multinet, adam_multinet.x_batch, adam_multinet.t_batch);

        for (idx=0;idx<hidden_layer_num+1;idx++) {
            // printf("idx:%d %d\n", multinet.all_size_list[idx], multinet.all_size_list[idx+1]);
            // SGD
            sgd_update(&sgd_optimizer, sgd_multinet.W[idx], sgd_multinet.gW[idx], sgd_multinet.all_size_list[idx] * sgd_multinet.all_size_list[idx+1]);
            sgd_update(&sgd_optimizer, sgd_multinet.b[idx], sgd_multinet.gb[idx], sgd_multinet.all_size_list[idx+1]);

            memcpy(sgd_multinet.layers.Affine[idx].W, sgd_multinet.W[idx], sizeof(double) * sgd_multinet.all_size_list[idx] * sgd_multinet.all_size_list[idx+1]);
            memcpy(sgd_multinet.layers.Affine[idx].b, sgd_multinet.b[idx], sizeof(double) * sgd_multinet.all_size_list[idx+1]);

            // Momentum
            momentum_update(&momentum_optimizer, momentum_multinet.W[idx], momentum_multinet.gW[idx], vW[idx], momentum_multinet.all_size_list[idx] * momentum_multinet.all_size_list[idx+1]);
            momentum_update(&momentum_optimizer, momentum_multinet.b[idx], momentum_multinet.gb[idx], vb[idx], momentum_multinet.all_size_list[idx+1]);

            memcpy(momentum_multinet.layers.Affine[idx].W, momentum_multinet.W[idx], sizeof(double) * momentum_multinet.all_size_list[idx] * momentum_multinet.all_size_list[idx+1]);
            memcpy(momentum_multinet.layers.Affine[idx].b, momentum_multinet.b[idx], sizeof(double) * momentum_multinet.all_size_list[idx+1]);


            // AdaGrad
            adagrad_update(&adagrad_optimizer, adagrad_multinet.W[idx], adagrad_multinet.gW[idx], hW[idx], adagrad_multinet.all_size_list[idx] * adagrad_multinet.all_size_list[idx+1]);
            adagrad_update(&adagrad_optimizer, adagrad_multinet.b[idx], adagrad_multinet.gb[idx], hb[idx], adagrad_multinet.all_size_list[idx+1]);

            memcpy(adagrad_multinet.layers.Affine[idx].W, adagrad_multinet.W[idx], sizeof(double) * adagrad_multinet.all_size_list[idx] * adagrad_multinet.all_size_list[idx+1]);
            memcpy(adagrad_multinet.layers.Affine[idx].b, adagrad_multinet.b[idx], sizeof(double) * adagrad_multinet.all_size_list[idx+1]);

            // Adam
            adam_update(&adam_optimizer, adam_multinet.W[idx], adam_multinet.gW[idx], mW[idx], vvW[idx], adam_multinet.all_size_list[idx] * adam_multinet.all_size_list[idx+1]);
            adam_update(&adam_optimizer, adam_multinet.b[idx], adam_multinet.gb[idx], mb[idx], vvW[idx], adam_multinet.all_size_list[idx+1]);

            memcpy(adam_multinet.layers.Affine[idx].W, adam_multinet.W[idx], sizeof(double) * adam_multinet.all_size_list[idx] * adam_multinet.all_size_list[idx+1]);
            memcpy(adam_multinet.layers.Affine[idx].b, adam_multinet.b[idx], sizeof(double) * adam_multinet.all_size_list[idx+1]);

        }

        double sgd_loss_ret = 0.0;
        double momentum_loss_ret = 0.0;
        double adagrad_loss_ret = 0.0;
        double adam_loss_ret = 0.0;
        loss(&sgd_multinet, &sgd_loss_ret, sgd_multinet.x_batch, sgd_multinet.t_batch);
        loss(&momentum_multinet, &momentum_loss_ret, momentum_multinet.x_batch, momentum_multinet.t_batch);
        loss(&adagrad_multinet, &adagrad_loss_ret, adagrad_multinet.x_batch, adagrad_multinet.t_batch);
        loss(&adam_multinet, &adam_loss_ret, adam_multinet.x_batch, adam_multinet.t_batch);

        if (i%100 == 0) {
            printf("iters_num: %d\n", i);
            printf("SGD: %f\n", sgd_loss_ret);
            printf("Momentum: %f\n", momentum_loss_ret);
            printf("AdaGrad: %f\n", adagrad_loss_ret);
            printf("Adam: %f\n", adam_loss_ret);
        }

        sgd_train_loss[i] = sgd_loss_ret;
        momentum_train_loss[i] = momentum_loss_ret;
        adagrad_train_loss[i] = adagrad_loss_ret;
        adam_train_loss[i] = adam_loss_ret;
        iters_num_arr[i] = i;

    }

    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set multiplot\n");
    fprintf(gp, "set grid\n");

    plot_graph_f(&gp, iters_num_arr, sgd_train_loss, iters_num);
    plot_graph_f(&gp, iters_num_arr, momentum_train_loss, iters_num);
    plot_graph_f(&gp, iters_num_arr, adagrad_train_loss, iters_num);
    plot_graph_f(&gp, iters_num_arr, adam_train_loss, iters_num);

    fprintf(gp, "set nomultiplot\n");
    fprintf(gp, "exit\n");


    multilayer_free(&sgd_multinet);
    multilayer_free(&momentum_multinet);
    multilayer_free(&adagrad_multinet);
    multilayer_free(&adam_multinet);

    free(x_train);
    free(t_train);
    free(x_test);
    free(t_test);
    free(sgd_train_loss);
    free(momentum_train_loss);
    free(adagrad_train_loss);
    free(adam_train_loss);
    free(iters_num_arr);
    free(batch_mask);

    for (idx=0;idx<hidden_layer_num+1;idx++) {
         free(vW[idx]);
         free(vb[idx]);
    }
    free(vW);
    free(vb);

    for (idx=0;idx<hidden_layer_num+1;idx++) {
         free(hW[idx]);
         free(hb[idx]);
    }
    free(hW);
    free(hb);

    for (idx=0;idx<hidden_layer_num+1;idx++) {
         free(mW[idx]);
         free(mb[idx]);
    }
    free(mW);
    free(mb);

    for (idx=0;idx<hidden_layer_num+1;idx++) {
         free(vvW[idx]);
         free(vvb[idx]);
    }
    free(vvW);
    free(vvb);

    return 0;
}
