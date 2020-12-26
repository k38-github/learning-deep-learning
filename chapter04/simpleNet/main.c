#include <stdio.h>
#include <stdlib.h>
#include "simpleNet.h"
#include "../../common/function.h"

int main(void) {
    simpleNet net;

    init(&net);

    print_matrix(net.W, 2, 3, "f");
    printf("\n");

    double X[2] = {0.6, 0.9};
    double Y[3] = {0};

    predict(&net, Y, X, 1);
    print_matrix(Y, 1, 3, "f");
    printf("\n");

    free(net.W);

    return 0;
}
