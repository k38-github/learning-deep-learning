#ifndef _TWOLAYERNET
#define _TWOLAYERNET

typedef struct TwoLayerNet {
    double *W1;
    double *b1;
    double *W2;
    double *b2;
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;
    double weight_init_std;
    double *gW1;
    double *gb1;
    double *gW2;
    double *gb2;
    double *x_batch;
    double *t_batch;
}TwoLayerNet;

int init(TwoLayerNet *, int, int, int, int, double);
int predict(TwoLayerNet *, double *, double *);
int loss(TwoLayerNet *, double *, double *, int);
int accuracy(TwoLayerNet *);
//int numerical_gradient_all(TwoLayerNet *, double *, double *);

#endif
