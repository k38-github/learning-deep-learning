#include <stdio.h>
#include <stdlib.h>
#include "MulLayer.h"
#include "../../common/function.h"

int init(MulLayer *this) {
    this->x = 0.0;
    this->y = 0.0;

    return 0;
}

int forward(MulLayer *this, double *out, double x, double y) {
    this->x = x;
    this->y = y;

    *out = x * y;

    return 0;
}

int backward(MulLayer *this, double *dx, double *dy, double dout) {
    *dx = dout * this->y;
    *dy = dout * this->x;

    return 0;
}
