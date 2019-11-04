#ifndef WENO_H
#define WENO_H

#include "../../etc/types.h"

Mat weno_launcher(Vecr ub, iVecr nX, int N, int V);

#endif // WENO_H
