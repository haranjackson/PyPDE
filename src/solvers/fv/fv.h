#ifndef FV_H
#define FV_H

#include "../../etc/types.h"

void finite_volume(void (*F)(double *, double *, int),
                   void (*B)(double *, double *, int),
                   void (*S)(double *, double *), Matr u, Matr qh, iVecr nX,
                   double dt, Vecr dX, int FLUX, int N);

#endif // FV_H
