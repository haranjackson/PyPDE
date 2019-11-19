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

DGSolver::DGSolver(void (*_F)(double *, double *, int),
                   void (*_B)(double *, double *, int),
                   void (*_S)(double *, double *), Vecr _dX, bool _STIFF,
                   int _N, int _V)
    : F(_F), B(_B), S(_S), dX(_dX), STIFF(_STIFF), N(_N), V(_V) {

  ndim = dX.size();
  Nd = pow(N, ndim);

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

  Mat M(N * Nd, V);
  Mat dM(Nd, V);

  std::vector<Mat> f(ndim);
  std::vector<Mat> df(ndim);

  for (int d = 0; d < ndim; d++) {
    f[d] = M;
    df[d] = dM;
  }

  std::vector<Mat> dq = df;

  Mat b(V, V);

  iVec indsInner = iVec::Zero(ndim);

  if (F != NULL) {
    for (int ind = 0; ind < N * Nd; ind++)
      for (int d = 0; d < ndim; d++) {
        F(f[d].row(ind).data(), q.row(ind).data(), d);
      }
  }

  for (int t = 0; t < N; t++) {

    for (int d = 0; d < ndim; d++) {

      derivs(dq[d], q.block(t * Nd, 0, Nd, V), d, DERVALS, ndim, dX);

      if (F != NULL) {
        derivs(df[d], f[d].block(t * Nd, 0, Nd, V), d, DERVALS, ndim, dX);
      }
    }

    for (int idx = 0; idx < Nd; idx++) {

      int ind = t * Nd + idx;

      if (S == NULL)
        ret.row(ind).setZero();
      else
        S(ret.row(ind).data(), q.row(ind).data());

      double c = WGHTS(t);
      for (int d = 0; d < ndim; d++) {

        if (B != NULL) {
          B(b.data(), q.row(ind).data(), d);
          ret.row(ind) -= b * dq[d].row(idx);
        }

        if (F != NULL)
          ret.row(ind) -= df[d].row(idx);

        c *= WGHTS(indsInner(d));
      }
      ret.row(ind) *= c;

      update_inds(indsInner, N);
    }
  }

  ret *= dt;
  ret += Ww;
  return ret;
}

Vec DGSolver::obj(Vecr q, Matr Ww, double dt) {

  MatMap qmat(q.data(), N * Nd, V, OuterStride(V));
  Mat tmp = rhs(qmat, Ww, dt);

  iVec indsInner = iVec::Zero(ndim);

  for (int t = 0; t < N; t++)
    for (int k = 0; k < N; k++) {

      int idx = 0;
      int indt = t * Nd;
      int indk = k * Nd;
      while (idx < Nd) {

        double c = DG_MAT(t, k);
        for (int d = 0; d < ndim; d++)
          c *= WGHTS(indsInner(d));

        tmp.row(indt + idx) -= c * qmat.row(indk + idx);

        update_inds(indsInner, N);
        idx++;
      }
    }

  VecMap ret(tmp.data(), tmp.rows() * tmp.cols());
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

Mat DGSolver::predictor(Matr wh, double dt) {

  Mat qh(wh.rows() * N, V);

  Mat Ww(N * Nd, V);
  Mat q0(N * Nd, V);

  for (int ind = 0; ind < wh.rows(); ind += Nd) {

    MatMap wi(wh.data() + ind * V, Nd, V, OuterStride(V));

    initial_condition(Ww, wi);
    initial_guess(q0, wi);

    if (STIFF) {

      using std::placeholders::_1;
      VecFunc obj_bound = std::bind(&DGSolver::obj, this, _1, Ww, dt);

      VecMap q0v(q0.data(), q0.rows() * q0.cols());

      Vec res = nonlin_solve(obj_bound, q0v, DG_TOL);

      Mat resMat(MatMap(res.data(), N * Nd, V, OuterStride(V)));

      qh.block(ind * N, 0, N * Nd, V) = resMat;

    } else {

      bool failed = true;
      for (int count = 0; count < DG_IT; count++) {

        Mat q1 = DG_U.solve(rhs(q0, Ww, dt));

        aMat absDiff = (q1 - q0).array().abs();

        if ((absDiff > DG_TOL * (1. + q0.array().abs())).any()) {
          q0 = q1;
          continue;
        } else {
          qh.block(ind * N, 0, N * Nd, V) = q1;
          failed = false;
          break;
        }
      }
      if (failed)
        std::cout << "DG iteration failed on ind=" << ind << "\n";
    }
  }
  return qh;
}
