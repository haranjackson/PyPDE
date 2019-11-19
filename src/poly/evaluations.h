#ifndef EVALUATIONS_H
#define EVALUATIONS_H

#include "../types.h"

void derivs(Matr ret, Matr qh, int d, Matr DERVALS, int ndim, Vecr dX);

void endpts(Matr ret, Matr qh, int d, int e, Matr ENDVALS, int ndim);

#endif // EVALUATIONS_H
