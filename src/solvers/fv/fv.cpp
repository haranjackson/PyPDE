#include "fv.h"
#include "../../poly/basis.h"
#include "../../poly/evaluations.h"
#include "../../types.h"
#include "fluxes.h"

#include <iostream>

FVSolver::FVSolver(void (*_F)(double *, double *, double *, int),
                   void (*_B)(double *, double *, int),
                   void (*_S)(double *, double *), iVecr _nX, Vecr _dX,
                   int _FLUX, int _N, int _V, bool _secondOrder)
    : F(_F), B(_B), S(_S), nX(_nX), dX(_dX), FLUX(_FLUX), N(_N), V(_V),
      secondOrder(_secondOrder) {

  nX1 = nX;
  nX2 = nX;
  nX1.array() += 1;
  nX2.array() += 2;

  ndim = nX.size();
  Nd = std::pow(N, ndim);
  Nd_ = std::pow(N, ndim - 1);

  NODES = scaled_nodes(N);
  WGHTS = scaled_weights(N);

  std::vector<poly> basis = basis_polys(N);
  DERVALS = derivative_values(basis, NODES);
  ENDVALS = end_values(basis);
}

void FVSolver::centers(Matr u, Matr qh, double dt) {

  Mat M(Nd, V);
  std::vector<Mat> dq(ndim);
  for (int d = 0; d < ndim; d++)
    dq[d] = M;

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
        derivs(dq[d], qhi, d, DERVALS, ndim, dX);

      for (int idx = 0; idx < Nd; idx++) {

        if (S == NULL)
          s.setZero();
        else
          S(s.data(), qhi.row(idx).data());

        if (B != NULL)
          for (int d = 0; d < ndim; d++) {
            B(b.data(), qhi.row(idx).data(), d);
            s -= b * dq[d].row(idx);
          }

        double tmp = dt * WGHTS(t);

        for (int i = 0; i < ndim; i++)
          tmp *= WGHTS(indsInner(i));

        u.row(uCell) += tmp * s;

        update_inds(indsInner, N);
      }
    }

    update_inds(indsOuter, nX);
  }
}

void FVSolver::qh_derivatives(Matr dqh, Matr qh) {

  for (int qhCell = 0; qhCell < nX2.prod(); qhCell++)
    for (int t = 0; t < N; t++) {

      int ind = (qhCell * N + t) * Nd * V;
      MatMap qh0(qh.data() + ind, Nd, V, OuterStride(V));

      for (int d = 0; d < ndim; d++) {
        MatMap dqhd(dqh.data() + d * qh.size() + ind, Nd, V, OuterStride(V));
        derivs(dqhd, qh0, d, DERVALS, ndim, dX);
      }
    }
}

void FVSolver::calculate_endpoints(Matr dqh, Matr qh0, Matr qh1, Matr q0,
                                   Matr q1, Matr dq0, Matr dq1, int ind0,
                                   int ind1, int d) {

  endpts(q0, qh0, d, 1, ENDVALS, ndim);
  endpts(q1, qh1, d, 0, ENDVALS, ndim);

  if (secondOrder)
    for (int d_ = 0; d_ < ndim; d_++) {

      int off = d_ * dqh.size() / ndim;
      MatMap dqh0(dqh.data() + off + ind0, Nd, V, OuterStride(V));
      MatMap dqh1(dqh.data() + off + ind1, Nd, V, OuterStride(V));

      MatMap dq0_(dq0.data() + d_ * V, Nd, V, OuterStride(ndim * V));
      MatMap dq1_(dq1.data() + d_ * V, Nd, V, OuterStride(ndim * V));

      endpts(dq0_, dqh0, d, 1, ENDVALS, ndim);
      endpts(dq1_, dqh1, d, 0, ENDVALS, ndim);
    }
}

void FVSolver::interfaces(Matr u, Matr qh, double dt) {

  FluxGenerator fluxGenerator(F, B, NODES, WGHTS, dX, V, FLUX, ndim);

  Vec f = Vec::Zero(V);
  Vec b = Vec::Zero(V);

  Mat q0(Nd_, V);
  Mat q1(Nd_, V);

  Mat dq0(Nd_ * ndim, V);
  Mat dq1(Nd_ * ndim, V);

  iVec indsOuter = iVec::Zero(ndim); // runs over all cells in ub
  iVec indsInner = iVec::Zero(ndim - 1);

  Mat dqh(ndim * qh.rows(), V);
  if (secondOrder)
    qh_derivatives(dqh, qh);

  Indexer indexer0(nX, nX2);
  Indexer indexer1(nX, nX2);

  // (i = 0 ... (nX+1); j = 0 ... (nY+1); etc
  for (int uCell = 0; uCell < nX1.prod(); uCell++) {

    indexer0.set_indices(indsOuter);

    for (int t = 0; t < N; t++) {

      int ind0 = (indexer0.qh * N + t) * Nd * V;
      MatMap qh0(qh.data() + ind0, Nd, V, OuterStride(V));

      for (int d = 0; d < ndim; d++) {

        double c = dt * WGHTS(t) / (2. * dX(d));

        indsOuter(d) += 1;
        indexer1.set_indices(indsOuter);
        indsOuter(d) -= 1;

        int ind1 = (indexer1.qh * N + t) * Nd * V;
        MatMap qh1(qh.data() + ind1, Nd, V, OuterStride(V));

        calculate_endpoints(dqh, qh0, qh1, q0, q1, dq0, dq1, ind0, ind1, d);

        for (int idx = 0; idx < Nd_; idx++) {

          // TODO: fix dq0, dq1
          if (F != NULL)
            fluxGenerator.flux(
                f, q0.row(idx), q1.row(idx), dq0.block(idx * ndim, 0, ndim, V),
                dq1.block(idx * ndim, 0, ndim, V), d, secondOrder);

          if (B != NULL)
            fluxGenerator.Bint(b, q0.row(idx), q1.row(idx), d);

          double tmp = c;
          for (int i = 0; i < ndim - 1; i++)
            tmp *= WGHTS(indsInner(i));

          if (indexer0.valid)
            u.row(indexer0.u) -= tmp * (b + f);

          if (indexer1.valid)
            u.row(indexer1.u) -= tmp * (b - f);

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
