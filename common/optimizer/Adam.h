#ifndef _ADAM
#define _ADAM

// Adam
typedef struct Adam {
    double lr;
    double beta1;
    double beta2;
    int    iter;
}Adam;

int adam_init(Adam *, double, double, double);
int adam_update(Adam *, double *, double *, double *, double *, int);

#endif
