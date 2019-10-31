#ifndef DG_H
#define DG_H

#include "../../etc/types.h"

void predictor(Vecr qh, Vecr wh, double dt, Vecr dX, int ncell, bool STIFF);

#endif // DG_H
