#include "dg.h"
#include "../../grid/indexing.h"
#include "../../poly/basis.h"
#include "../../poly/evaluations.h"
#include "../../scipy/algos/newton_krylov.h"
#include "../../types.h"
#include "dg_matrices.h"
#include <vector>

#include <iostream>

void initial_guess(Matr q, Matr w) {
  // Returns a Galerkin intial guess consisting of the value of q at t=0
  int N = q.size() / w.size();
  int ROWS = w.rows();
  int COLS = w.cols();

  for (int i = 0; i < N; i++)
    q.block(i * ROWS, 0, ROWS, COLS) = w;
}

DGSolver::DGSolver(void (*_F)(double *, double *, double *, int),
                   void (*_B)(double *, double *, int),
                   void (*_S)(double *, double *), Vecr _dX, bool _STIFF,
                   int _N, int _V)
    : F(_F), B(_B), S(_S), dX(_dX), STIFF(_STIFF), N(_N), V(_V) {

  ndim = dX.size();
  Nd = std::pow(N, ndim);

  std::vector<poly> basis = basis_polys(N);
  NODES = scaled_nodes(N);
  WGHTS = scaled_weights(N);

  DERVALS = derivative_values(basis, NODES);
  ENDVALS = end_values(basis);

  Mat DG_END = end_value_products(basis);
  Mat DG_DER = derivative_products(basis, NODES, WGHTS);
  DG_MAT = DG_END - DG_DER.transpose();

  std::vector<Mat> tmp(ndim + 1);
  tmp[0] = DG_MAT;
  for (int i = 0; i < ndim; i++)
    tmp[i + 1] = WGHTS.asDiagonal();

  DG_U = Dec(kron(tmp));
}

Mat DGSolver::rhs(Matr q, Matr Ww, double dt) {

  Mat ret(Nd * N, V);

  Mat dq(Nd * ndim, V);
  Mat f(Nd * ndim, V);
  Mat df(Nd * ndim, V);

  Mat b(V, V);
  Mat tmpDeriv(Nd, V);

  iVec indsInner = iVec::Zero(ndim);

  for (int t = 0; t < N; t++) {

    MatMap qt(q.data() + t * Nd * V, Nd, V, OuterStride(V));
    MatMap rett(ret.data() + t * Nd * V, Nd, V, OuterStride(V));

    for (int d = 0; d < ndim; d++) {

      // first 2 args of derivs must be contiguous
      derivs(tmpDeriv, qt, d, DERVALS, ndim, dX);

      MatMap dqMap(dq.data() + d * V, Nd, V, OuterStride(ndim * V));
      dqMap = tmpDeriv;
    }

    if (F != NULL) {

      for (int idx = 0; idx < Nd; idx++) {

        int ind = idx * ndim;

        for (int d = 0; d < ndim; d++)
          F(f.row(ind + d).data(), qt.row(idx).data(),
            dq.block(ind, 0, ndim, V).data(), d);
      }

      for (int d = 0; d < ndim; d++) {

        Mat fMap(MatMap(f.data() + d * V, Nd, V, OuterStride(ndim * V)));
        derivs(tmpDeriv, fMap, d, DERVALS, ndim, dX);

        MatMap dfMap(df.data() + d * V, Nd, V, OuterStride(ndim * V));
        dfMap = tmpDeriv;
      }
    }

    for (int idx = 0; idx < Nd; idx++) {

      if (S == NULL)
        rett.row(idx).setZero();
      else
        S(rett.row(idx).data(), qt.row(idx).data());

      double c = WGHTS(t);
      for (int d = 0; d < ndim; d++) {

        if (B != NULL) {
          B(b.data(), qt.row(idx).data(), d);
          rett.row(idx) -= b * dq.row(idx * ndim + d);
        }

        if (F != NULL)
          rett.row(idx) -= df.row(idx * ndim + d);

        c *= WGHTS(indsInner(d));
      }
      rett.row(idx) *= c;

      update_inds(indsInner, N);
    }
  }
  ret *= dt;
  ret += Ww;
  return ret;
}

Vec DGSolver::obj(Vecr qv, Matr Ww, double dt) {

  MatMap q(qv.data(), N * Nd, V, OuterStride(V));
  Mat tmp = rhs(q, Ww, dt);

  iVec indsInner = iVec::Zero(ndim);

  for (int t = 0; t < N; t++)
    for (int k = 0; k < N; k++) {

      int indt = t * Nd;
      int indk = k * Nd;

      for (int idx = 0; idx < Nd; idx++) {

        double c = DG_MAT(t, k);
        for (int d = 0; d < ndim; d++)
          c *= WGHTS(indsInner(d));

        tmp.row(indt + idx) -= c * q.row(indk + idx);

        update_inds(indsInner, N);
      }
    }
  VecMap ret(tmp.data(), tmp.size());
  return ret;
}

void DGSolver::initial_condition(Matr Ww, Matr w) {

  iVec indsInner = iVec::Zero(ndim);

  for (int t = 0; t < N; t++)
    for (int idx = 0; idx < Nd; idx++) {

      double c = ENDVALS(0, t);
      for (int d = 0; d < ndim; d++)
        c *= WGHTS(indsInner(d));

      Ww.row(t * Nd + idx) = c * w.row(idx);

      update_inds(indsInner, N);
    }
}

Mat DGSolver::stiff_solve(Matr q0, Matr Ww, double dt) {

  using std::placeholders::_1;
  VecFunc obj_bound = std::bind(&DGSolver::obj, this, _1, Ww, dt);

  VecMap q0v(q0.data(), q0.rows() * q0.cols());

  Vec res = nonlin_solve(obj_bound, q0v, DG_TOL);

  Mat resMat(MatMap(res.data(), N * Nd, V, OuterStride(V)));

  return resMat;
}

Mat DGSolver::nonstiff_solve(Matr q0, Matr Ww, double dt) {

  Mat q1(N * Nd, V);
  for (int count = 0; count < DG_IT; count++) {

    q1 = DG_U.solve(rhs(q0, Ww, dt));

    aMat absDiff = (q1 - q0).array().abs();

    if ((absDiff > DG_TOL * (1. + q0.array().abs())).any())
      q0 = q1;
    else
      break;
  }
  return q1;
}

Mat DGSolver::predictor(Matr wh, double dt) {

  Mat qh(wh.rows() * N, V);

  Mat Ww(N * Nd, V);
  Mat q0(N * Nd, V);

  for (int ind = 0; ind < wh.rows(); ind += Nd) {

    MatMap wi(wh.data() + ind * V, Nd, V, OuterStride(V));

    initial_condition(Ww, wi);
    initial_guess(q0, wi);

    if (STIFF)
      qh.block(ind * N, 0, N * Nd, V) = stiff_solve(q0, Ww, dt);
    else
      qh.block(ind * N, 0, N * Nd, V) = nonstiff_solve(q0, Ww, dt);
  }
  return qh;
}
