#include "solvers/iterator.h"
#include "solvers/weno/weno.h"
#include "types.h"

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

extern "C" void weno_solver(double *ret, double *_u, int *_nX, int ndim, int N,
                            int V) {

  iVecMap nX(_nX, ndim);

  WenoSolver wenoSolver(nX, N, V);

  int ncell = nX.prod();
  MatMap u(_u, ncell, V, OuterStride(V));

  Mat wh = wenoSolver.reconstruction(u);

  VecMap whVec(wh.data(), wh.size());

  for (int i = 0; i < whVec.size(); i++)
    ret[i] = whVec(i);
}