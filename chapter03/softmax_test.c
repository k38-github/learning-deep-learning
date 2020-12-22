#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(void) {
    double *X, *Y;

    X = (double *)malloc(sizeof(double)*3);
    Y = (double *)malloc(sizeof(double)*3);

    X[0] = 0.3; X[1] = 2.9; X[2] = 4.0;
    softmax_function(X, Y, 3);
    print_matrix(Y, 1, 3, "f");
    printf("\n");

    double Y1;
    sum_function(Y, &Y1, 3);
    printf("%11.8f\n", Y1);

    X[0] = 1010.0; X[1] = 1000.0; X[2] = 990.0;
    softmax_measures_function(X, Y, 3);
    print_matrix(Y, 1, 3, "e");
    printf("\n");

    free(X);
    free(Y);

    return 0;
}
