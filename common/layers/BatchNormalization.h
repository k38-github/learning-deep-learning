#ifndef _BATCHNORMALIZATION
#define _BATCHNORMALIZATION

typedef struct BatchNormalization {
    double *gamma;
    double *beta;
    double momentum;
    double *running_mean;
    double *running_var;
    int batch_size;
    double *xc;
    double *std;
    double *dgamma;
    double *dbeta;
    int col_size;
    int row_size;
} BatchNormalization;

int batchnornalization_init(BatchNormalization *, double, int, int);
int batchnornalization_free(BatchNormalization *);
int batchnornalization_forward(BatchNormalization *, double *, double *, char *);
int batchnornalization_backward(BatchNormalization *, double *, double *);

#endif
