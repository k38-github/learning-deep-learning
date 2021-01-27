#ifndef _MOMENTUM
#define _MOMENTUM

// Momentum SGD
typedef struct Momentum {
    double lr;
    double momentum;
}Momentum;

int momentum_init(Momentum *, double, double);
int momentum_update(Momentum *, double *, double *, double *, int);

#endif
