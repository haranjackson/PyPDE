#ifndef ITERATOR_H
#define ITERATOR_H

#include "../types.h"

void iterator(void (*F)(double *, double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr u, double tf, iVecr nX, aVecr dX, double CFL,
              iVecr boundaryTypes, bool STIFF, int FLUX, int N, int ndt,
              bool secondOrder, Matr ret, int nThreads);

#endif