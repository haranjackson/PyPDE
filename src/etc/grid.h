#ifndef GRID_H
#define GRID_H

#include "types.h"

void boundaries(Vecr u, Vecr ub, iVecr nX, iVecr boundaryTypes, int N, int V);

int extended_dimensions(iVecr nX, int ext);

#endif // GRID_H
