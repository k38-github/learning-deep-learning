#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(int argc, char *argv[]) {

    double *a, *b;
    int i;
    int j;
    int n = 2;
    int m = 3;

    a = (double *)malloc(sizeof(double)*n*m);
    b = (double *)malloc(sizeof(double)*n*m);

    a[0] = 5.0; a[1] = 6.0; a[2] = 7.0;
    a[3] = 8.0; a[4] = 9.0; a[5] = 10.0;

    trans_function(b, a, n, m);
    print_matrix(b, m, n, "f");
    printf("\n");

    double axis_row_sum = 0.0;
    double tmp[3] = {0};
    double ret[3] = {0};

    for (i=0;i<m;i++) {
        for (j=0;j<n;j++) {
            tmp[j] = b[j+(n*i)];
        }
        sum_function(tmp, &axis_row_sum, n);
        ret[i] = axis_row_sum;
        axis_row_sum = 0.0;
    }
    print_matrix(ret, 1, m, "f");
    printf("\n");

    double sigmoid_ret[3] = {0};
    sigmoid_grad_function(ret, sigmoid_ret, 3);
    print_matrix(sigmoid_ret, 1, m, "f");
    printf("\n");

    free(a);
    free(b);

    return 0 ;
}
