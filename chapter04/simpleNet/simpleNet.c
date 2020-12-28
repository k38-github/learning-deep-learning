#include <stdio.h>
#include <stdlib.h>
#include "simpleNet.h"
#include "../../common/function.h"

int init(simpleNet *this, double *x, int x_col, int x_row, double *t, int t_col, int t_row) {
    int i;

    this->x = (double *)malloc(sizeof(double) * x_col * x_row);
    for (i=0;i<x_col*x_row;i++) {
        this->x[i] = x[i];
    }
    this->x_col = x_col;
    this->x_row = x_row;

    this->t = (double *)malloc(sizeof(double) * t_col * t_row);
    for (i=0;i<t_col*t_row;i++) {
        this->t[i] = t[i];
    }
    this->t_col = t_col;
    this->t_row = t_row;

    this->W = (double *)malloc(sizeof(double) * x_row * t_row);
    random_randn(this->W, this->x_row, this->t_row);

    return 0;
}

int predict(simpleNet *this, double *y, double *x) {

    dot_function(&y, x, this->W, this->x_col, this->x_row, this->t_row);

    return 0;
}

int loss(simpleNet *this, double *ret, double *w, int e) {

    double *z;
    z = (double *)malloc(sizeof(double) * this->t_row);

    // z(1x3) = x(1x2) x W(2x3)
    predict(this, z, this->x);
    printf("predict\n");
    print_matrix(z, 1, 3, "f");
    printf("\n");

    double *y;
    y = (double *)malloc(sizeof(double) * this->t_row);

    softmax_measures_function(z, y, 3);
    printf("softmax\n");
    print_matrix(y, 1, 3, "f");
    printf("\n");

    cross_entropy_error(y, this->t, ret , this->t_row);
    printf("cross_entropy_error: %12.8f\n", *ret);

    free(z);
    free(y);

}
