#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {

    double x[2] = {3.0, 4.0};
    int element = 2;
    double ret[2] ={0};

    numerical_gradient(function_2, x, element, ret);

    int i;
    for (i=0;i<element;i++) {
        printf("%12.8f\n", ret[i]);
    }

    double x1[2] = {0.0, 2.0};
    numerical_gradient(function_2, x1, element, ret);
    for (i=0;i<element;i++) {
        printf("%12.8f\n", ret[i]);
    }

    double x2[2] = {3.0, 0.0};
    numerical_gradient(function_2, x2, element, ret);
    for (i=0;i<element;i++) {
        printf("%12.8f\n", ret[i]);
    }


    return 0;
}
