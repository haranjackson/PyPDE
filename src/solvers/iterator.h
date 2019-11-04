#ifndef ITERATOR_H
#define ITERATOR_H

#include "../etc/types.h"

double timestep(Matr u, aVecr dX, double CFL, double t, double tf, int count,
                int V);

std::vector<Vec> iterator(Matr u, double tf, iVecr nX, aVecr dX, double CFL,
                          iVecr boundaryTypes, bool STIFF, int FLUX, int N);

#endif // ITERATOR_H
