#ifndef _SGD
#define _SGD

// 確率的勾配降下法(Stochastic Gradient Descent)
typedef struct SGD {
    double lr;
}SGD;

int sgd_init(SGD *, double);
int sgd_update(SGD *, double *, double *, int);

#endif
