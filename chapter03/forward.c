#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int forward(double *X, double *Y4) {
    // 入力層から第1層目への信号の伝達
    double *W, *Y, *B1, *A1, *Z1;
    int n = 1;
    int m = 2;
    int l = 3;

    //X = (double *)malloc(sizeof(double)*n*m);
    W = (double *)malloc(sizeof(double)*m*l);
    Y = (double *)malloc(sizeof(double)*n*l);
    B1 = (double *)malloc(sizeof(double)*n*l);
    A1 = (double *)malloc(sizeof(double)*n*l);
    Z1 = (double *)malloc(sizeof(double)*n*l);

    // 1x2
    X[0] = 1.0; X[1] = 0.5;
    printf("X\n");
    print_matrix(X, n, m, "f");
    printf("\n");

    // 2x3
    W[0] = 0.1; W[1] = 0.3; W[2] = 0.5;
    W[3] = 0.2; W[4] = 0.4; W[5] = 0.6;
    printf("W\n");
    print_matrix(W, m, l, "f");
    printf("\n");

    // 1x3
    // Y = X dot W
    dot_function(&Y, X, W, n, m, l);
    printf("Y = X dot W\n");
    print_matrix(Y, n, l, "f");
    printf("\n");

    // 1x3
    B1[0] = 0.1; B1[1] = 0.2; B1[2] = 0.3;
    printf("B1\n");
    print_matrix(B1, n, l, "f");
    printf("\n");

    // 1x3
    // A1 = Y + B1
    matrix_sum(&A1, Y, B1, n, l);
    printf("A1 = Y + B1\n");
    print_matrix(A1, n, l, "f");
    printf("\n");

    // 1x3
    // Z1 = sigmoid(A1)
    sigmoid_function(A1, Z1, l);
    printf("Z1 = sigmoid(A1)\n");
    print_matrix(Z1, n, l, "f");
    printf("\n");

    // 第1層目から第2層目への信号の伝達
    double *W2, *Y2, *B2, *A2, *Z2;

    W2 = (double *)malloc(sizeof(double)*l*m);
    Y2 = (double *)malloc(sizeof(double)*n*m);
    B2 = (double *)malloc(sizeof(double)*n*m);
    A2 = (double *)malloc(sizeof(double)*n*m);
    Z2 = (double *)malloc(sizeof(double)*n*m);

    // 3x2
    W2[0] = 0.1; W2[1] = 0.4;
    W2[2] = 0.2; W2[3] = 0.5;
    W2[4] = 0.3; W2[5] = 0.6;
    printf("W2\n");
    print_matrix(W2, l, m, "f");
    printf("\n");

    // 1x2
    // Y2 = Z1 dot W2
    dot_function(&Y2, Z1, W2, n, l, m);
    printf("Y2 = Z1 dot W2\n");
    print_matrix(Y2, n, m, "f");
    printf("\n");

    // 1x2
    printf("B2\n");
    B2[0] = 0.1; B2[1] = 0.2;
    print_matrix(B2, n, m, "f");
    printf("\n");

    // 1x2
    // A2 = Y2 + B2
    matrix_sum(&A2, Y2, B2, n, m);
    printf("A2 = Y2 + B2\n");
    print_matrix(A2, n, m, "f");
    printf("\n");

    // 1x2
    // Z2 = sigmoid(A2)
    sigmoid_function(A2, Z2, m);
    printf("Z2 = sigmoid(A2)\n");
    print_matrix(Z2, n, m, "f");
    printf("\n");

    // 第2層目から出力層への信号の伝達
    double *W3, *Y3, *B3, *A3;

    W3 = (double *)malloc(sizeof(double)*m*m);
    Y3 = (double *)malloc(sizeof(double)*n*m);
    B3 = (double *)malloc(sizeof(double)*n*m);
    A3 = (double *)malloc(sizeof(double)*n*m);
    //Y4 = (double *)malloc(sizeof(double)*n*m);

    // 2x2
    W3[0] = 0.1; W3[1] = 0.3;
    W3[2] = 0.2; W3[3] = 0.4;
    printf("W3\n");
    print_matrix(W3, m, m, "f");
    printf("\n");

    // 1x2
    // Y3 = Z2 dot W3
    dot_function(&Y3, Z2, W3, n, m, m);
    printf("Y3 = Z2 dot W3\n");
    print_matrix(Y3, n, m, "f");
    printf("\n");

    // 1x2
    B3[0] = 0.1; B3[1] = 0.2;
    printf("B3\n");
    print_matrix(B3, n, m, "f");
    printf("\n");

    // 1x2
    // A3 = Y3 + B3
    matrix_sum(&A3, Y3, B3, n, m);
    printf("A3 = Y3 + B3\n");
    print_matrix(A3, n, m, "f");
    printf("\n");

    // 1x2
    // Y4 = identity_functin(A3)
    identity_function(A3, Y4, m);
    printf("Y4 = identity_function(A3)\n");
    print_matrix(Y4, n, m, "f");
    printf("\n");

    //free(X);
    free(W);
    free(Y);
    free(B1);
    free(A1);
    free(Z1);

    free(W2);
    free(Y2);
    free(B2);
    free(A2);
    free(Z2);

    free(W3);
    free(Y3);
    free(B3);
    free(A3);
    //free(Y4);

    return 0;

}
