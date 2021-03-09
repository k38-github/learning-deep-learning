#ifndef _BATCHNORMALIZATION
#define _BATCHNORMALIZATION

typedef struct BatchNormalization {
    double *gamma;
    double *beta;
    double momentum;
    double *running_mean;
    double *running_var;
    double *xc;
    double *xn;
    double *std;
    double *dgamma;
    double *dbeta;
    int col_size;
    int row_size;
} BatchNormalization;

int batchnormalization_init(BatchNormalization *, double *, double *, double, double *, double *, int, int);
int batchnornalization_free(BatchNormalization *);
int batchnornalization_forward(BatchNormalization *, double *, double *, char *);
int batchnornalization_backward(BatchNormalization *, double *, double *);

#endif
