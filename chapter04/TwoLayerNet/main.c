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
    int size[4];

    load_mnist(&X_TRAIN, &T_TRAIN, &X_TEST, &T_TEST, size);

    double *x_train;
    x_train = (double *)malloc(sizeof(double) * size[0]);
    normalize(X_TRAIN, x_train, size[0]);

    int *t_train;
    int one_hot_value = 10;
    t_train = (int *)malloc(sizeof(int) * size[1] * one_hot_value);
    one_hot(T_TRAIN, t_train, size[1]);

    double *train_loss_list;

    //TwoLayerNet net;
    int iters_num = 10000;
    int train_size = size[0];
    double learning_rate = 0.1;

    int input_size = 784;
    int hidden_size = 100;
    int output_size = 10;
    int batch_size = 100;
    double weight_init_std = 0.01;

    init(&net, input_size, hidden_size, output_size, batch_size, weight_init_std);

    // print_matrix(net.W1, 784, 100, "f");
    // printf("\n");
    // print_matrix(net.b1, 1, 100, "f");
    // printf("\n");
    // print_matrix(net.W2, 100, 10, "f");
    // printf("\n");
    // print_matrix(net.b2, 1, 10, "f");
    // printf("\n");

    int *batch_mask;
    net.x_batch = (double *)malloc(sizeof(double) * input_size * batch_size);
    net.t_batch = (double *)malloc(sizeof(double) * output_size * batch_size);
    batch_mask = (int *)malloc(sizeof(int) * batch_size);

    int i, j, k, l, m;
    for (i=0;i<iters_num;i++) {
        printf("iters_num: %d\n", i);
        random_choice(train_size, input_size, batch_size, batch_mask);

        l = 0;
        for (j=0;j<batch_size;j++) {
            //printf("x: batch_size: %d batch_mask: %d\n", j, batch_mask[j]);

            for (k=(batch_mask[j]*input_size)+input_size*j;k<(batch_mask[j]*input_size)+input_size*(j+1);k++) {
                net.x_batch[l] = x_train[k];

                //printf("%3.2f", net.x_batch[l]);
                //if (l%28 == 0) {
                //    printf("\n");
                //}

                l++;
            }
        }

        m = 0;
        for (j=0;j<batch_size;j++) {
            //printf("t: batch_size: %d batch_mask: %d\n", j, batch_mask[j]);
            for (k=(batch_mask[j]*output_size)+output_size*j;k<(batch_mask[j]*output_size)+output_size*(j+1);k++) {
                net.t_batch[m] = t_train[k];

                //printf("%f\n", net.t_batch[m]);

                m++;
            }
        }

        fflush(stdout);

        //numerical_gradient_all(&net, x_batch, t_batch);
        numerical_gradient(loss_W, net.W1, net.input_size * net.hidden_size, net.gW1);
        numerical_gradient(loss_W, net.b1, net.hidden_size, net.gb1);
        numerical_gradient(loss_W, net.W2, net.hidden_size * net.output_size, net.gW2);
        numerical_gradient(loss_W, net.b2, net.output_size, net.gb2);


        for (j=0;j<net.input_size * net.hidden_size;j++) {
            net.W1[j] -= learning_rate * net.gW1[j];
        }
        for (j=0;j<net.hidden_size;j++) {
            net.b1[j] -= learning_rate * net.gb1[j];
        }
        for (j=0;j<net.hidden_size * net.output_size;j++) {
            net.W2[j] -= learning_rate * net.gW2[j];
        }
        for (j=0;j<net.output_size;j++) {
            net.b2[j] -= learning_rate * net.gb2[j];
        }

        accuracy(&net);

    }


    return 0;
}

double loss_W(double *w, int e) {
    double ret = 0.0;

    loss(&net, &ret, w, e);

    return ret;
}


