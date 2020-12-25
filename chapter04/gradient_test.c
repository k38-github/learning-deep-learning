#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {

    double init_x[2] = {-3.0, 4.0};
    int element = sizeof(init_x)/sizeof(double);
    double lr = 0.1;
    int step_num = 100;
    double ret[2] = {0};

    gradient_descent(function_2, init_x, element, lr, step_num, ret);
    printf("%e %e\n", ret[0], ret[1]);

    lr = 10.0;
    gradient_descent(function_2, init_x, element, lr, step_num, ret);
    printf("%e %e\n", ret[0], ret[1]);

    lr = pow(1, -10);
    gradient_descent(function_2, init_x, element, lr, step_num, ret);
    printf("%e %e\n", ret[0], ret[1]);

    return 0;
}
