#ifndef _RMSPROP
#define _RMSPROP

// RMSprop
typedef struct RMSprop {
    double lr;
    double decay_rate;
}RMSprop;

int rmsprop_init(RMSprop *, double, double);
int rmsprop_update(RMSprop *, double *, double *, double *, int);

#endif
