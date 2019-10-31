#ifndef FLUXES_H
#define FLUXES_H

#include "../../etc/types.h"

const int RUSANOV = 0;
const int ROE = 1;
const int OSHER = 2;

Vec Bint(void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
         Vecr NODES, Vecr WGHTS);

Vec D_OSH(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
          Vecr NODES, Vecr WGHTS);

Vec D_ROE(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
          Vecr NODES, Vecr WGHTS);

Vec D_RUS(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d);

#endif // FLUXES_H
