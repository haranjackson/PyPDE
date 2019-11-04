#ifndef WENO_H
#define WENO_H

#include "../../etc/types.h"

Mat weno_reconstruction(Matr ub, iVecr nX, int N, int V);

#endif // WENO_H
