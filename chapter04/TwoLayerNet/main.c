#include <stdio.h>
#include <stdlib.h>
#include "TwoLayerNet.h"
#include "../../common/function.h"

int main(void) {
    TwoLayerNet net;
    int input_size = 784;
    int hidden_size = 100;
    int output_size = 10;
    int batch_size = 100;
    double weight_init_std = 0.01;

    init(&net, input_size, hidden_size, output_size, batch_size, weight_init_std);
    //print_matrix(net.W1, 784, 100, "f");
    //printf("\n");

    //print_matrix(net.b1, 1, 100, "f");
    //printf("\n");

    //print_matrix(net.W2, 100, 10, "f");
    //printf("\n");

    //print_matrix(net.b2, 1, 10, "f");
    //printf("\n");

    double *x;
    double *y;

    x = (double *)malloc(sizeof(double) * batch_size * input_size);
    y = (double *)malloc(sizeof(double) * batch_size * output_size);

    predict(&net, y, x);
    print_matrix(y, batch_size, output_size, "e");
    printf("\n");




    return 0;
}
