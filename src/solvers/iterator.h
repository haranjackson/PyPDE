#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/system.h"
#include "../etc/types.h"

double timestep(void (*F)(double *, double *, int),
                void (*B)(double *, double *, int), Matr u, aVecr dX,
                double CFL, double t, double tf, int count);

void iterator(void (*F)(double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr u, double tf, iVecr nX, aVecr dX, double CFL,
              iVecr boundaryTypes, bool STIFF, int FLUX, int N, int ndt,
              Matr ret);

#endif