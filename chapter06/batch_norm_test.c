#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/function.h"
#include "../common/layers/AffineLayer.h"
#include "../common/layers/ReluLayer.h"
#include "../common/layers/SigmoidLayer.h"
#include "../common/layers/SoftmaxWithLossLayer.h"
#include "../common/layers/MultiLayerNetExtend.h"
#include "../common/optimizer/SGD.h"
#include "../dataset/mnist.h"

int main(void) {

    MultiLayerNetExtend multinet_extend;

    char *X_TRAIN;
    char *T_TRAIN;
    char *X_TEST;
    char *T_TEST;
    int size[4] = {0};
    int one_hot_value = 10;
    int data_num = 0;
    int data_size = 1000;

    load_mnist(&X_TRAIN, &T_TRAIN, &X_TEST, &T_TEST, size);

    double *_x_train;
    _x_train = (double *)malloc(sizeof(double) * size[0]);
    normalize(X_TRAIN, _x_train, size[0]);

    double *x_train;
    x_train = (double *)malloc(sizeof(double) * 784 * 1000);

    for (data_num=0;data_num<784*data_size;data_num++) {
        x_train[data_num] = _x_train[data_num];
    }

    int *_t_train;
    _t_train = (int *)malloc(sizeof(int) * size[1] * one_hot_value);
    one_hot(T_TRAIN, _t_train, size[1]);

    int *t_train;
    t_train = (int *)malloc(sizeof(int) * size[1] * one_hot_value);

    for (data_num=0;data_num<data_size;data_num++) {
        t_train[data_num] = _t_train[data_num];
    }

    double *x_test;
    x_test = (double *)malloc(sizeof(double) * size[2]);
    normalize(X_TEST, x_test, size[2]);

    int *t_test;
    t_test = (int *)malloc(sizeof(int) * size[3] * one_hot_value);
    one_hot(T_TEST, t_test, size[3]);

    int max_epochs = 20;
    int epoch_count = 0;
    int iters_num = 1000;
    int train_size = 784*data_size;
    double *sgd_train_loss;
    double *iters_num_arr;

    sgd_train_loss = (double *)malloc(sizeof(double) * iters_num);
    iters_num_arr = (double *)malloc(sizeof(double) * iters_num);

    int input_size = 784;
    int hidden_size_list[5] = {100, 100, 100, 100, 100};
    int hidden_layer_num = 5;
    int output_size = 10;
    int batch_size = 128;

    char *activation = "relu";
    char *weight_init_std = "relu";
    double weight_decay_lambda = 0.0;
    char *use_dropout = "false";
    double dropout_ration = 0.0;
    char *use_batchnorm = "true";

    multilayerextend_init(&multinet_extend, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, weight_init_std, weight_decay_lambda, use_dropout, dropout_ration, use_batchnorm);

    int idx = 0;
    // SGD
    double learning_rate = 0.01;
    SGD sgd_optimizer;
    sgd_init(&sgd_optimizer, learning_rate);

    int *batch_mask;
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    multinet_extend.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    multinet_extend.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        // printf("iters_num: %d\n", i);

        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        m = 0;
        k = 0;

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*input_size;k<(batch_mask[j]*input_size)+input_size;k++) {
                multinet_extend.x_batch[l] = x_train[k];
                l++;
            }
        }

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {
                multinet_extend.t_batch[m] = t_train[k];
                m++;
            }
        }

        gradient(&multinet_extend, multinet_extend.x_batch, multinet_extend.t_batch);

        for (idx=0;idx<hidden_layer_num+1;idx++) {
            // printf("idx:%d %d\n", multinet.all_size_list[idx], multinet.all_size_list[idx+1]);
            // SGD
            sgd_update(&sgd_optimizer, multinet_extend.W[idx], multinet_extend.gW[idx], multinet_extend.all_size_list[idx] * multinet_extend.all_size_list[idx+1]);
            sgd_update(&sgd_optimizer, multinet_extend.b[idx], multinet_extend.gb[idx], multinet_extend.all_size_list[idx+1]);

            memcpy(multinet_extend.layers.Affine[idx].W, multinet_extend.W[idx], sizeof(double) * multinet_extend.all_size_list[idx] * multinet_extend.all_size_list[idx+1]);
            memcpy(multinet_extend.layers.Affine[idx].b, multinet_extend.b[idx], sizeof(double) * multinet_extend.all_size_list[idx+1]);
        }

        double sgd_loss_ret = 0.0;
        char *train_flg = "false";
        loss(&multinet_extend, &sgd_loss_ret, multinet_extend.x_batch, multinet_extend.t_batch, train_flg);

        if (i%10 == 0) {
            printf("iters_num: %d\n", i);
            printf("SGD: %f\n", sgd_loss_ret);

            epoch_count++;
            if (epoch_count >= max_epochs) {
                break;
            }
        }

        sgd_train_loss[i] = sgd_loss_ret;
        iters_num_arr[i] = i;

    }
//
//    FILE *gp;
//    gp = popen("gnuplot -persist", "w");
//    fprintf(gp, "set multiplot\n");
//    fprintf(gp, "set grid\n");
//
//    plot_graph_f(&gp, iters_num_arr, sgd_train_loss, iters_num);
//    plot_graph_f(&gp, iters_num_arr, momentum_train_loss, iters_num);
//    plot_graph_f(&gp, iters_num_arr, adagrad_train_loss, iters_num);
//    plot_graph_f(&gp, iters_num_arr, adam_train_loss, iters_num);
//
//    fprintf(gp, "set nomultiplot\n");
//    fprintf(gp, "exit\n");
//
//
    multilayerextend_free(&multinet_extend);

    free(_x_train);
    free(_t_train);
    free(x_train);
    free(t_train);
    free(x_test);
    free(t_test);
    free(sgd_train_loss);
    free(iters_num_arr);
    free(batch_mask);

    return 0;
}
