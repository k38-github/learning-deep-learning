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
}TwoLayerNet;

int init(TwoLayerNet *, int, int, int, int, double);
int predict(TwoLayerNet *, double *, double *);
int loss(TwoLayerNet *, double *, double *, double *);
//accuracy();
//numerical_gradient();

#endif
