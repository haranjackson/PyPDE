#include "fv.h"
#include "../../etc/indexing.h"
#include "../../etc/types.h"
#include "../poly/basis.h"
#include "../poly/evaluations.h"
#include "fluxes.h"

#include <iostream>

bool in_bounds(iVecr inds, iVecr bounds, int offset) {
  for (int i = 0; i < inds.size(); i++) {
    if (inds(i) + offset < 0 || inds(i) + offset >= bounds(i))
      return false;
  }
  return true;
}

FVSolver::FVSolver(void (*_F)(double *, double *, int),
                   void (*_B)(double *, double *, int),
                   void (*_S)(double *, double *), iVecr _nX, Vecr _dX,
                   int _FLUX, int _N, int _V)
    : F(_F), B(_B), S(_S), nX(_nX), dX(_dX), FLUX(_FLUX), N(_N), V(_V) {

  nX1 = nX;
  nX2 = nX;
  nX1.array() += 1;
  nX2.array() += 2;

  ndim = nX.size();
  Nd = pow(N, ndim);
  Nd_ = pow(N, ndim - 1);

  NODES = scaled_nodes(N);
  WGHTS = scaled_weights(N);

  std::vector<poly> basis = basis_polys(N);
  DERVALS = derivative_values(basis, NODES);
  ENDVALS = end_values(basis);
}

void FVSolver::centers(Matr u, Matr qh, double dt) {

  Mat dqh_dx(Nd, V);
  std::vector<Mat> dq(ndim);
  for (int d = 0; d < ndim; d++)
    dq[d] = dqh_dx;

  Mat b(V, V);
  Vec s(V);

  iVec indsOuter = iVec::Zero(ndim);
  iVec indsInner = iVec::Zero(ndim);

  for (int uCell = 0; uCell < nX.prod(); uCell++) {

    int recCell = index(indsOuter, nX2, 1);

    for (int t = 0; t < N; t++) {

      int tCell = (recCell * N + t) * Nd * V;

      MatMap qhi(qh.data() + tCell, Nd, V, OuterStride(V));

      for (int d = 0; d < ndim; d++)
        derivs(dq[d], qhi, d, DERVALS, ndim);

      int idx = 0;
      while (idx < Nd) {

        if (S == NULL)
          s.setZero();
        else
          S(s.data(), qhi.row(idx).data());

        if (B != NULL)
          for (int d = 0; d < ndim; d++) {
            B(b.data(), qhi.row(idx).data(), d);
            s -= b * dq[d].row(idx) / dX(d);
          }

        double tmp = dt * WGHTS(t);

        for (int i = 0; i < ndim; i++)
          tmp *= WGHTS(indsInner(i));

        u.row(uCell) += tmp * s;

        update_inds(indsInner, N);
        idx++;
      }
    }

    update_inds(indsOuter, nX);
  }
}

void FVSolver::interfaces(Matr u, Matr qh, double dt) {

  FluxGenerator fluxGenerator(F, B, NODES, WGHTS, V, FLUX);

  Vec f(V);
  Vec b = Vec::Zero(V);

  Mat q0(Nd_, V);
  Mat q1(Nd_, V);

  iVec indsOuter = iVec::Zero(ndim); // runs over all cells in ub
  iVec indsInner = iVec::Zero(ndim - 1);

  // (i = 0 ... (nX+1); j = 0 ... (nY+1); etc
  for (int uCell = 0; uCell < nX1.prod(); uCell++) {

    int uind0 = index(indsOuter, nX, -1);  // (i-1, j-2, ...)
    int ubind0 = index(indsOuter, nX2, 0); // (i, j, ...)
    bool inBounds0 = in_bounds(indsOuter, nX, -1);

    for (int t = 0; t < N; t++) {

      int ind0 = (ubind0 * N + t) * Nd * V;
      MatMap qh0(qh.data() + ind0, Nd, V, OuterStride(V));

      for (int d = 0; d < ndim; d++) {

        double c = dt * WGHTS(t) / (2. * dX(d));

        indsOuter(d) += 1;
        int uind1 = index(indsOuter, nX, -1);
        int ubind1 = index(indsOuter, nX2, 0);
        bool inBounds1 = in_bounds(indsOuter, nX, -1);
        indsOuter(d) -= 1;

        int ind1 = (ubind1 * N + t) * Nd * V;
        MatMap qh1(qh.data() + ind1, Nd, V, OuterStride(V));

        endpts(q0, qh0, d, 1, ENDVALS, ndim);
        endpts(q1, qh1, d, 0, ENDVALS, ndim);

        for (int idx = 0; idx < Nd_; idx++) {

          fluxGenerator.flux(f, q0.row(idx), q1.row(idx), d);

          if (B != NULL)
            fluxGenerator.Bint(b, q0.row(idx), q1.row(idx), d);

          double tmp = c;
          for (int i = 0; i < ndim - 1; i++)
            tmp *= WGHTS(indsInner(i));

          if (inBounds0)
            u.row(uind0) -= tmp * (b + f);
          if (inBounds1)
            u.row(uind1) -= tmp * (b - f);

          update_inds(indsInner, N);
        }
      }
    }

    update_inds(indsOuter, nX1);
  }
}

void FVSolver::apply(Matr u, Matr qh, double dt) {

  if (B != NULL || S != NULL)
    centers(u, qh, dt);

  if (F != NULL || B != NULL)
    interfaces(u, qh, dt);
}
