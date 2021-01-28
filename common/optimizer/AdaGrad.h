#ifndef _ADAGRAD
#define _ADAGRAD

// AdaGrad
typedef struct AdaGrad {
    double lr;
}AdaGrad;

int adagrad_init(AdaGrad *, double);
int adagrad_update(AdaGrad *, double *, double *, double *, int);

#endif
