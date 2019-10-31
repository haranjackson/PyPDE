#ifndef FV_H
#define FV_H

#include "../../etc/types.h"

void fv_launcher(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, int FLUX);

#endif // FV_H
