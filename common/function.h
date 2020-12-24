#ifndef _FUNCTION
#define _FUNCTION

#include <stdio.h>

int random_choice(int, int, int, int *);
int min_function(double *, double *, int);
int max_function(double *, double *, int);
int sum_function(double *, double *, int);
int matrix_sum(double **, double *, double *, int, int);
int dot_function(double **, double *, double *, int, int, int);
int print_matrix(double *, int, int, char *);
int step_function(double *, double *, int);
int sigmoid_function(double *, double *, int);
int relu_function(double *, double *, int);
int identity_function(double *, double *, int);
int softmax_function(double *, double *, int);
int softmax_measures_function(double *, double *, int);
int mean_squared_error(double *, double *, double *, int);
int cross_entropy_error(double *, double *, double *, int);
int numerical_diff(double (*)(double), double, double *);
double function_1(double);
double function_tmp1(double);
double function_tmp2(double);
int tangent_line(double (*)(double), double, double, double *);
double function_2(double *, int);
int numerical_gradient(double (*)(double *, int), double *, int, double *);
int array_range(double, double, double, double *);
int plot_graph(double *, double *, int);
int plot_graph_f(FILE **, double *, double *, int);

#endif
