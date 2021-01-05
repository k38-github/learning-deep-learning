#include <stdio.h>
#include <stdlib.h>
#include "TwoLayerNet.h"
#include "../../common/function.h"
#include "../../dataset/mnist.h"

double loss_W(double *, int);

TwoLayerNet net;

int main(void) {
    char *X_TRAIN;
    char *T_TRAIN;
    char *X_TEST;
    char *T_TEST;
    int size[4] = {0};

    load_mnist(&X_TRAIN, &T_TRAIN, &X_TEST, &T_TEST, size);

    double *x_train;
    x_train = (double *)malloc(sizeof(double) * size[0]);
    normalize(X_TRAIN, x_train, size[0]);

    int *t_train;
    int one_hot_value = 10;
    t_train = (int *)malloc(sizeof(int) * size[1] * one_hot_value);
    one_hot(T_TRAIN, t_train, size[1]);

    free(X_TRAIN);
    free(T_TRAIN);
    free(X_TEST);
    free(T_TEST);

    int iters_num = 10000;
    int train_size = size[0];
    double learning_rate = 0.1;

    int input_size = 784;
    int hidden_size = 50;
    int output_size = 10;
    int batch_size = 100;
    double weight_init_std = 0.01;

    init(&net, input_size, hidden_size, output_size, batch_size, weight_init_std);

    double *train_loss;
    double *iters_num_arr;
    train_loss = (double *)malloc(sizeof(double) * iters_num);
    iters_num_arr = (double *)malloc(sizeof(double) * iters_num);

    int *batch_mask;
    net.x_batch = (double *)malloc(sizeof(double) * input_size * batch_size);
    net.t_batch = (double *)malloc(sizeof(double) * output_size * batch_size);
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        printf("iters_num: %d\n", i);
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
        }

        for (j=0;j<batch_size;j++) {
            //printf("t: batch_size: %d batch_mask: %d\n", j, batch_mask[j]);
            for (k=batch_mask[j]*output_size;k<(batch_mask[j]*output_size)+output_size;k++) {

                //printf("%f\n", net.t_batch[m]);

                net.t_batch[m] = t_train[k];
                m++;
            }
        }

        gradient(&net, net.x_batch, net.t_batch);
        //numerical_gradient(loss_W, net.W1, net.input_size * net.hidden_size, net.gW1);
        //numerical_gradient(loss_W, net.b1, net.batch_size * net.hidden_size, net.gb1);
        //numerical_gradient(loss_W, net.W2, net.hidden_size * net.output_size, net.gW2);
        //numerical_gradient(loss_W, net.b2, net.batch_size * net.output_size, net.gb2);

        for (j=0;j<net.input_size * net.hidden_size;j++) {
            net.W1[j] -= learning_rate * net.gW1[j];
        }
        for (j=0;j<net.batch_size * net.hidden_size;j++) {
            net.b1[j] -= learning_rate * net.gb1[j];
        }
        for (j=0;j<net.hidden_size * net.output_size;j++) {
            net.W2[j] -= learning_rate * net.gW2[j];
        }
        for (j=0;j<net.batch_size * net.output_size;j++) {
            net.b2[j] -= learning_rate * net.gb2[j];
        }

        double ret = 0.0;
        loss(&net, &ret, x_train, batch_size);
        printf("cross_entropy: %.18f\n", ret);
        fflush(stdout);

        iters_num_arr[i] = i;
        train_loss[i] = ret;
        //accuracy(&net);

    }

    plot_graph(iters_num_arr, train_loss, iters_num);

    free(train_loss);
    free(batch_mask);
    free(x_train);
    free(t_train);

    free(net.W1);
    free(net.b1);
    free(net.W2);
    free(net.b2);

    free(net.gW1);
    free(net.gb1);
    free(net.gW2);
    free(net.gb2);

    return 0;
}

double loss_W(double *w, int e) {
    double ret = 0.0;

    loss(&net, &ret, w, e);

    return ret;
}

