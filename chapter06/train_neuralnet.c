#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "TwoLayerNet.h"
#include "../common/function.h"
#include "../common/optimizer/SGD.h"
#include "../common/optimizer/Momentum.h"
#include "../common/optimizer/Nesterov.h"
#include "../common/optimizer/AdaGrad.h"
#include "../common/optimizer/RMSprop.h"
#include "../common/optimizer/Adam.h"
#include "../dataset/mnist.h"

TwoLayerNet net;

int main(void) {
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

    int iters_num = 10000;
    int train_size = size[0];

    int input_size = 784;
    int hidden_size = 50;
    int output_size = 10;
    int batch_size = 100;
    double weight_init_std = 0.01;

    init(&net, input_size, hidden_size, output_size, batch_size, weight_init_std);

    // SGD
    // double learning_rate = 0.1;
    // SGD optimizer;
    // sgd_init(&optimizer, learning_rate);

    // Momentum
    // double learning_rate = 0.01;
    // double momentum = 0.9;
    // Momentum optimizer;
    // momentum_init(&optimizer, learning_rate, momentum);

    // double *vW1;
    // double *vb1;
    // double *vW2;
    // double *vb2;
    // vW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    // vb1 = (double *)calloc(hidden_size, sizeof(double));
    // vW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    // vb2 = (double *)calloc(output_size, sizeof(double));

    // Nesterov
    // double learning_rate = 0.01;
    // double momentum = 0.9;
    // Nesterov optimizer;
    // nesterov_init(&optimizer, learning_rate, momentum);

    // double *vW1;
    // double *vb1;
    // double *vW2;
    // double *vb2;
    // vW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    // vb1 = (double *)calloc(hidden_size, sizeof(double));
    // vW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    // vb2 = (double *)calloc(output_size, sizeof(double));

    // AdaGrad
    // double learning_rate = 0.01;
    // AdaGrad optimizer;
    // adagrad_init(&optimizer, learning_rate);

    // double *hW1;
    // double *hb1;
    // double *hW2;
    // double *hb2;
    // hW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    // hb1 = (double *)calloc(hidden_size, sizeof(double));
    // hW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    // hb2 = (double *)calloc(output_size, sizeof(double));

    // RMSprop
    // double learning_rate = 0.01;
    // double decay_rate = 0.99;
    // RMSprop optimizer;
    // rmsprop_init(&optimizer, learning_rate, decay_rate);

    // double *hW1;
    // double *hb1;
    // double *hW2;
    // double *hb2;
    // hW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    // hb1 = (double *)calloc(hidden_size, sizeof(double));
    // hW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    // hb2 = (double *)calloc(output_size, sizeof(double));

    // Adam
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    Adam optimizer;
    adam_init(&optimizer, learning_rate, beta1, beta2);

    double *mW1;
    double *mb1;
    double *mW2;
    double *mb2;
    mW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    mb1 = (double *)calloc(hidden_size, sizeof(double));
    mW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    mb2 = (double *)calloc(output_size, sizeof(double));

    double *vW1;
    double *vb1;
    double *vW2;
    double *vb2;
    vW1 = (double *)calloc(input_size * hidden_size, sizeof(double));
    vb1 = (double *)calloc(hidden_size, sizeof(double));
    vW2 = (double *)calloc(hidden_size * output_size, sizeof(double));
    vb2 = (double *)calloc(output_size, sizeof(double));

    int acc = 0;
    double *train_loss;
    double *iters_num_arr;
    double *train_acc;
    double *test_acc;
    double *acc_count;
    train_loss = (double *)malloc(sizeof(double) * iters_num);
    iters_num_arr = (double *)malloc(sizeof(double) * iters_num);
    train_acc = (double *)malloc(sizeof(double) * (iters_num / (train_size/input_size/batch_size)));
    test_acc = (double *)malloc(sizeof(double) * (iters_num / (train_size/input_size/batch_size)));
    acc_count = (double *)malloc(sizeof(double) * (iters_num / (train_size/input_size/batch_size)));

    int *batch_mask;
    net.x_batch = (double *)malloc(sizeof(double) * batch_size * input_size);
    net.t_batch = (double *)malloc(sizeof(double) * batch_size * output_size);
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {

        //printf("iters_num: %d\n", i);
        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        m = 0;
        k = 0;

        for (j=0;j<batch_size;j++) {
            //printf("x: batch_size: %d batch_mask: %d\n", j, batch_mask[j]);

            for (k=batch_mask[j]*input_size;k<(batch_mask[j]*input_size)+input_size;k++) {

                //printf("%3.2f", net.x_batch[l]);
                //if (l%28 == 0) {
                //    printf("\n");
                //}

                net.x_batch[l] = x_train[k];
                l++;
            }
            //printf("\n");
        }

        for (j=0;j<batch_size;j++) {
            //printf("t: batch_size: %d batch_mask: %d\n", j, batch_mask[j]);
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {

                //printf("%f\n", net.t_batch[m]);

                net.t_batch[m] = t_train[k];
                m++;
            }
            //printf("\n");
        }

        gradient(&net, net.x_batch, net.t_batch);

        // SGD
        // sgd_update(&optimizer, net.W1, net.gW1, net.input_size * net.hidden_size);
        // sgd_update(&optimizer, net.b1, net.gb1, net.hidden_size);
        // sgd_update(&optimizer, net.W2, net.gW2, net.hidden_size * net.output_size);
        // sgd_update(&optimizer, net.b2, net.gb2, net.output_size);

        // Momentum
        // momentum_update(&optimizer, net.W1, net.gW1, vW1, net.input_size * net.hidden_size);
        // momentum_update(&optimizer, net.b1, net.gb1, vb1, net.hidden_size);
        // momentum_update(&optimizer, net.W2, net.gW2, vW2, net.hidden_size * net.output_size);
        // momentum_update(&optimizer, net.b2, net.gb2, vb2, net.output_size);

        // Nesterov
        // nesterov_update(&optimizer, net.W1, net.gW1, vW1, net.input_size * net.hidden_size);
        // nesterov_update(&optimizer, net.b1, net.gb1, vb1, net.hidden_size);
        // nesterov_update(&optimizer, net.W2, net.gW2, vW2, net.hidden_size * net.output_size);
        // nesterov_update(&optimizer, net.b2, net.gb2, vb2, net.output_size);

        // AdaGrad
        // adagrad_update(&optimizer, net.W1, net.gW1, hW1, net.input_size * net.hidden_size);
        // adagrad_update(&optimizer, net.b1, net.gb1, hb1, net.hidden_size);
        // adagrad_update(&optimizer, net.W2, net.gW2, hW2, net.hidden_size * net.output_size);
        // adagrad_update(&optimizer, net.b2, net.gb2, hb2, net.output_size);

        // RMSprop
        // rmsprop_update(&optimizer, net.W1, net.gW1, hW1, net.input_size * net.hidden_size);
        // rmsprop_update(&optimizer, net.b1, net.gb1, hb1, net.hidden_size);
        // rmsprop_update(&optimizer, net.W2, net.gW2, hW2, net.hidden_size * net.output_size);
        // rmsprop_update(&optimizer, net.b2, net.gb2, hb2, net.output_size);

        //RMSprop
        adam_update(&optimizer, net.W1, net.gW1, mW1, vW1, net.input_size * net.hidden_size);
        adam_update(&optimizer, net.b1, net.gb1, mb1, vb1, net.hidden_size);
        adam_update(&optimizer, net.W2, net.gW2, mW2, vW2, net.hidden_size * net.output_size);
        adam_update(&optimizer, net.b2, net.gb2, mb2, vb2, net.output_size);

        memcpy(net.layers.Affine1.W, net.W1, sizeof(double) * net.input_size * net.hidden_size);
        memcpy(net.layers.Affine1.b, net.b1, sizeof(double) * net.hidden_size);
        memcpy(net.layers.Affine2.W, net.W2, sizeof(double) * net.hidden_size * net.output_size);
        memcpy(net.layers.Affine2.b, net.b2, sizeof(double) * net.output_size);

        double ret = 0.0;
        loss(&net, &ret, net.x_batch, net.t_batch);
        printf("cross_entropy: %.18f\n", ret);
        fflush(stdout);

        train_loss[i] = ret;
        iters_num_arr[i] = i;

        // iters_num%600 == 0
        if (i%(train_size/input_size/batch_size) == 0) {
            net.batch_size = size[0]/input_size;
            net.layers.Relu1.size = net.batch_size * net.hidden_size;
            accuracy(&net, &train_acc[acc], x_train, t_train);

            net.batch_size = size[2]/input_size;
            net.layers.Relu1.size = net.batch_size * net.hidden_size;
            accuracy(&net, &test_acc[acc], x_test, t_test);

            printf("train acc, test acc | %f, %f\n", train_acc[acc], test_acc[acc]);
            acc_count[acc] = acc;
            acc++;
        }

        net.batch_size = batch_size;
        net.layers.Relu1.size = net.batch_size * net.hidden_size;

    }

    plot_graph(iters_num_arr, train_loss, iters_num);
    plot_graph(acc_count, train_acc, acc);
    plot_graph(acc_count, test_acc, acc);

    free(X_TRAIN);
    free(T_TRAIN);
    free(X_TEST);
    free(T_TEST);

    free(net.W1);
    free(net.b1);
    free(net.W2);
    free(net.b2);

    free(net.gW1);
    free(net.gb1);
    free(net.gW2);
    free(net.gb2);

    free(x_train);
    free(t_train);
    free(x_test);
    free(t_test);

    free(train_loss);
    free(iters_num_arr);
    free(train_acc);
    free(test_acc);
    free(acc_count);
    free(batch_mask);

    free(mW1);
    free(mb1);
    free(mW2);
    free(mb2);

    free(vW1);
    free(vb1);
    free(vW2);
    free(vb2);

    //free(hW1);
    //free(hb1);
    //free(hW2);
    //free(hb2);


    layers_free(&net);

    return 0;
}
