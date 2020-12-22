#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../common/function.h"

int main(int argc, char *argv[]) {

    double min = atof(argv[1]);
    double max = atof(argv[2]);
    double step = atof(argv[3]);
    double *in;
    double *out;

    int element = (fabs(min)+fabs(max))/step;

    in = malloc(sizeof(double) * element);
    out = malloc(sizeof(double) * element);

    array_range(min, max, step, in);

    step_function(in, out, element);
    plot_graph(in, out, element);

    sigmoid_function(in, out, element);
    plot_graph(in, out, element);

    relu_function(in, out, element);
    plot_graph(in, out, element);

    free(in);
    free(out);

    return 0;
}

