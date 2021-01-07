#ifndef _MULLAYER
#define _MULLAYER

typedef struct MulLayer {
    double x;
    double y;
}MulLayer;

int init(MulLayer *);
int forward(MulLayer *, double *, double, double);
int backward(MulLayer *, double *, double *, double);

#endif
