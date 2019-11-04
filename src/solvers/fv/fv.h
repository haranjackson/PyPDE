#ifndef FV_H
#define FV_H

#include "../../etc/types.h"

void fv_launcher(void (*F)(double *, double *, int),
                 void (*B)(double *, double *, int),
                 void (*S)(double *, double *), Vecr u, Matr qh, iVecr nX,
                 double dt, Vecr dX, int FLUX, int N);

#endif // FV_H
