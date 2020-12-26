#include <stdio.h>
#include <stdlib.h>
#include "simpleNet.h"
#include "../../common/function.h"

int init(simpleNet *this) {
    this->n = 2;
    this->m = 3;
    this->W = (double *)malloc(sizeof(double) * this->n*this->m);
    random_randn(this->W, this->n, this->m);

    return 0;
}

int predict(simpleNet *this, double *Y, double *X, int x_col) {

    printf("%d %d %d\n", x_col, this->n, this->m);
    dot_function(&Y, X, this->W, x_col, this->n, this->m);

    return 0;
}
