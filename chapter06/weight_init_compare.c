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
#include "../dataset/mnist.h"

int main(void) {

    MultiLayerNet std_multinet;
    MultiLayerNet xavier_multinet;
    MultiLayerNet he_multinet;

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
    double *std_train_loss;
    double *xavier_train_loss;
    double *he_train_loss;
    double *iters_num_arr;

    std_train_loss = (double *)malloc(sizeof(double) * iters_num);
    xavier_train_loss = (double *)malloc(sizeof(double) * iters_num);
    he_train_loss = (double *)malloc(sizeof(double) * iters_num);
    iters_num_arr = (double *)malloc(sizeof(double) * iters_num);

    int input_size = 784;
    int hidden_size_list[4] = {100, 100, 100, 100};
    int hidden_layer_num = 4;
    int output_size = 10;
    int batch_size = 128;

    char *activation = "relu";
    char *std_weight_init_std = "none";
    char *xavier_weight_init_std = "sigmoid";
    char *he_weight_init_std = "relu";

    double weight_decay_lambda = 0.0;

    multilayer_init(&std_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, std_weight_init_std, weight_decay_lambda);
    multilayer_init(&xavier_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, xavier_weight_init_std, weight_decay_lambda);
    multilayer_init(&he_multinet, input_size, hidden_size_list, hidden_layer_num, output_size, batch_size, activation, he_weight_init_std, weight_decay_lambda);

    int idx = 0;
    // SGD
    double learning_rate = 0.01;
    SGD std_sgd_optimizer;
    sgd_init(&std_sgd_optimizer, learning_rate);
    SGD xavier_sgd_optimizer;
    sgd_init(&xavier_sgd_optimizer, learning_rate);
    SGD he_sgd_optimizer;
    sgd_init(&he_sgd_optimizer, learning_rate);

    int *batch_mask;
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    std_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    std_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    xavier_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    xavier_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    he_multinet.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    he_multinet.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        // printf("iters_num: %d\n", i);

        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        m = 0;
        k = 0;

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*input_size;k<(batch_mask[j]*input_size)+input_size;k++) {
                std_multinet.x_batch[l] = x_train[k];
                xavier_multinet.x_batch[l] = x_train[k];
                he_multinet.x_batch[l] = x_train[k];
                l++;
            }
        }

        for (j=0;j<batch_size;j++) {
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {
                std_multinet.t_batch[m] = t_train[k];
                xavier_multinet.t_batch[m] = t_train[k];
                he_multinet.t_batch[m] = t_train[k];
                m++;
            }
        }

        gradient(&std_multinet, std_multinet.x_batch, std_multinet.t_batch);
        gradient(&xavier_multinet, xavier_multinet.x_batch, xavier_multinet.t_batch);
        gradient(&he_multinet, he_multinet.x_batch, he_multinet.t_batch);

        for (idx=0;idx<hidden_layer_num+1;idx++) {
            // printf("idx:%d %d\n", multinet.all_size_list[idx], multinet.all_size_list[idx+1]);
            // SGD
            sgd_update(&std_sgd_optimizer, std_multinet.W[idx], std_multinet.gW[idx], std_multinet.all_size_list[idx] * std_multinet.all_size_list[idx+1]);
            sgd_update(&std_sgd_optimizer, std_multinet.b[idx], std_multinet.gb[idx], std_multinet.all_size_list[idx+1]);

            memcpy(std_multinet.layers.Affine[idx].W, std_multinet.W[idx], sizeof(double) * std_multinet.all_size_list[idx] * std_multinet.all_size_list[idx+1]);
            memcpy(std_multinet.layers.Affine[idx].b, std_multinet.b[idx], sizeof(double) * std_multinet.all_size_list[idx+1]);

            sgd_update(&xavier_sgd_optimizer, xavier_multinet.W[idx], xavier_multinet.gW[idx], xavier_multinet.all_size_list[idx] * xavier_multinet.all_size_list[idx+1]);
            sgd_update(&xavier_sgd_optimizer, xavier_multinet.b[idx], xavier_multinet.gb[idx], xavier_multinet.all_size_list[idx+1]);

            memcpy(xavier_multinet.layers.Affine[idx].W, xavier_multinet.W[idx], sizeof(double) * xavier_multinet.all_size_list[idx] * xavier_multinet.all_size_list[idx+1]);
            memcpy(xavier_multinet.layers.Affine[idx].b, xavier_multinet.b[idx], sizeof(double) * xavier_multinet.all_size_list[idx+1]);

            sgd_update(&he_sgd_optimizer, he_multinet.W[idx], he_multinet.gW[idx], he_multinet.all_size_list[idx] * he_multinet.all_size_list[idx+1]);
            sgd_update(&he_sgd_optimizer, he_multinet.b[idx], he_multinet.gb[idx], he_multinet.all_size_list[idx+1]);

            memcpy(he_multinet.layers.Affine[idx].W, he_multinet.W[idx], sizeof(double) * he_multinet.all_size_list[idx] * he_multinet.all_size_list[idx+1]);
            memcpy(he_multinet.layers.Affine[idx].b, he_multinet.b[idx], sizeof(double) * he_multinet.all_size_list[idx+1]);
        }

        double std_loss_ret = 0.0;
        double xavier_loss_ret = 0.0;
        double he_loss_ret = 0.0;
        loss(&std_multinet, &std_loss_ret, std_multinet.x_batch, std_multinet.t_batch);
        loss(&xavier_multinet, &xavier_loss_ret, xavier_multinet.x_batch, xavier_multinet.t_batch);
        loss(&he_multinet, &he_loss_ret, he_multinet.x_batch, he_multinet.t_batch);

        if (i%100 == 0) {
            printf("iters_num: %d\n", i);
            printf("std: %f\n", std_loss_ret);
            printf("xavier: %f\n", xavier_loss_ret);
            printf("he: %f\n", he_loss_ret);
        }

        std_train_loss[i] = std_loss_ret;
        xavier_train_loss[i] = xavier_loss_ret;
        he_train_loss[i] = he_loss_ret;
        iters_num_arr[i] = i;

    }

    FILE *gp;
    gp = popen("gnuplot -persist", "w");
    fprintf(gp, "set multiplot\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set xrange [0.0:2000.0]\n");
    fprintf(gp, "set yrange [0.0:2.5]\n");

    plot_graph_f(&gp, iters_num_arr, std_train_loss, iters_num);
    plot_graph_f(&gp, iters_num_arr, xavier_train_loss, iters_num);
    plot_graph_f(&gp, iters_num_arr, he_train_loss, iters_num);

    fprintf(gp, "set nomultiplot\n");
    fprintf(gp, "exit\n");

    multilayer_free(&std_multinet);
    multilayer_free(&xavier_multinet);
    multilayer_free(&he_multinet);

    free(x_train);
    free(t_train);
    free(x_test);
    free(t_test);
    free(std_train_loss);
    free(xavier_train_loss);
    free(he_train_loss);
    free(iters_num_arr);
    free(batch_mask);

    return 0;
}
