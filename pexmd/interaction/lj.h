#ifndef LJ_H
#define LJ_H

#include "math.h"
float pair_force(float rsq, float ljf1, float ljf2);
float pair_energ(float r2inv, float r6inv, float lje1, float lje2);
float forces(float *x, long int* pairs, long int npairs, float eps,
             float sigma, float rcutsq, float *force);
#endif
