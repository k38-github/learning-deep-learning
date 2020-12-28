#include <stdio.h>
#include <stdlib.h>
#include "simpleNet.h"
#include "../../common/function.h"

double s_function(double *, int);

simpleNet net;

int main(void) {
    double x[2] = {0.6, 0.9};
    double t[3] = {0.0, 0.0, 1.0};

    //W(2x3)の要素数
    double *y;
    y = (double *)malloc(sizeof(net.x_row*net.t_row)/sizeof(double));

    init(&net, x, 1, 2, t, 1, 3);
    //net.W[0] = -0.37479063; net.W[1] = 1.57609041; net.W[2] = -1.32114857;
    //net.W[3] = -0.53829903; net.W[4] = -0.76678277; net.W[5] = -0.392584;

    print_matrix(net.W, 2, 3, "f");
    printf("\n");

    numerical_gradient(s_function, net.W, net.x_row*net.t_row, y);

    // y(1x3)
    printf("gradient\n");
    print_matrix(y, net.x_row, net.t_row, "f");
    printf("\n");

    // dW: [[ 0.14050792  0.36869773 -0.50920565]
    //      [ 0.21076189  0.55304659 -0.76380848]]

    free(net.W);
    free(y);

    return 0;
}

double s_function(double *w, int e) {
    double ret = 0.0;

    loss(&net, &ret, w, e);

    return ret;
}


