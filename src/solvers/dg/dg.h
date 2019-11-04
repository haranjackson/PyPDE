#ifndef DG_H
#define DG_H

#include "../../etc/types.h"

Mat predictor(void (*F)(double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr wh, double dt, Vecr dX, bool STIFF, int N, int V);

#endif // DG_H
