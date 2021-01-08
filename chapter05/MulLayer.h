#ifndef _MULLAYER
#define _MULLAYER

typedef struct MulLayer {
    double x;
    double y;
}MulLayer;

int mullayer_init(MulLayer *);
int mullayer_forward(MulLayer *, double *, double, double);
int mullayer_backward(MulLayer *, double *, double *, double);

#endif
