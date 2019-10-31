#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/types.h"

double timestep(Vecr u, aVecr dX, double CFL, double t, double tf, int count,
                int V);

std::vector<Vec> iterator(Vecr u, double tf, iVecr nX, aVecr dX, double CFL,
                          iVecr boundaryTypes, bool STIFF, int FLUX, int N);

#endif // ITERATOR_H
