#ifndef SYSTEM_H
#define SYSTEM_H

#include "../types.h"

Mat system_matrix(void (*F)(double *, double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, Matr dq, int d);

double max_abs_eigs(void (*F)(double *, double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, Matr dq, int d);

double max_abs_eigs_second_order(void (*F)(double *, double *, double *, int),
                                 Vecr q, Matr dq, int d, int N, Vecr dX);

#endif
