#ifndef SYSTEM_H
#define SYSTEM_H

#include "../types.h"

Mat system_matrix(void (*F)(double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, int d);

double max_abs_eigs(void (*F)(double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, int d);

#endif
