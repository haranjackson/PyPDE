#include "etc/types.h"
#include "solvers/iterator.h"

extern "C" void ader_solver(void (*F)(double *, double *, int),
                            void (*B)(double *, double *, int),
                            void (*S)(double *, double *), bool useF, bool useB,
                            bool useS, double *_u, double tf, int *_nX,
                            int ndim, double *_dX, double CFL,
                            int *_boundaryTypes, bool STIFF, int FLUX, int N,
                            int V, int ndt, double *_ret) {

  iVecMap nX(_nX, ndim);
  VecMap dX(_dX, ndim);
  iVecMap boundaryTypes(_boundaryTypes, ndim);

  int ncell = nX.prod();
  MatMap u(_u, ncell, V, OuterStride(V));
  MatMap ret(_ret, ndt, ncell * V, OuterStride(ncell * V));

  if (not useF)
    F = NULL;
  if (not useB)
    B = NULL;
  if (not useS)
    S = NULL;

  iterator(F, B, S, u, tf, nX, dX, CFL, boundaryTypes, STIFF, FLUX, N, ndt,
           ret);
}
