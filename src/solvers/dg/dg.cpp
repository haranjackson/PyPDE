#include "../../etc/indexing.h"
#include "../../etc/types.h"
#include "../../scipy/newton_krylov.h"
#include "../poly/basis.h"
#include "../poly/evaluations.h"
#include "dg_matrices.h"

const int DG_IT = 50;       // No. of iterations of non-Newton solver attempted
const double DG_TOL = 6e-6; // Convergence tolerance

void initial_guess(Matr q, Matr w) {
  // Returns a Galerkin intial guess consisting of the value of q at t=0
  int N = q.size() / w.size();
  int ROWS = w.rows();

  for (int i = 0; i < N; i++)
    for (int j = 0; j < ROWS; j++) {
      q.row(i * ROWS + j) = w.row(j);
    }
}

Mat rhs(void (*F)(double *, double *, int), void (*B)(double *, double *, int),
        void (*S)(double *, double *), Matr q, Matr Ww, double dt, Vecr dX,
        int N, int V, Matr DERVALS, Vecr WGHTS, Matr DG_DER) {

  int ndim = dX.size();
  int Nd = pow(N, ndim);

  Mat ret(Nd * N, V);

  Mat M = Mat::Zero(Nd, V);
  Mat M1 = Mat::Zero(N * Nd, V);

  std::vector<Mat> dq(ndim);
  std::vector<Mat> f(ndim);
  std::vector<Mat> df(ndim);

  for (int d = 0; d < ndim; d++) {
    f[d] = M1;
    df[d] = M;
    dq[d] = M;
  }

  Mat b(V, V);

  iVec indsInner = iVec::Zero(ndim);

  for (int ind = 0; ind < N * Nd; ind++)
    for (int d = 0; d < ndim; d++) {
      F(f[d].row(ind).data(), q.row(ind).data(), d);
    }

  for (int t = 0; t < N; t++) {

    for (int d = 0; d < ndim; d++) {

      derivs(dq[d], q.block(t * Nd, 0, Nd, V), d, DERVALS, ndim);
      dq[d] /= dX(d);

      derivs(df[d], f[d].block(t * Nd, 0, Nd, V), d, DERVALS, ndim);
      df[d] /= dX(d);
    }

    int idx = 0;
    while (idx < Nd) {

      int ind = t * Nd + idx;

      if (S == NULL)
        ret.row(ind).setZero();
      else
        S(ret.row(ind).data(), q.row(ind).data());

      double c = WGHTS(t);
      for (int d = 0; d < ndim; d++) {

        B(b.data(), q.row(ind).data(), d);
        ret.row(ind) -= b * dq[d].row(ind);

        ret.row(ind) -= df[d].row(ind);

        c *= WGHTS(indsInner(d));
      }
      ret.row(ind) *= c;

      update_inds(indsInner, N);
      idx++;
    }
  }

  return dt * ret + Ww;
}

Vec obj(void (*F)(double *, double *, int), void (*B)(double *, double *, int),
        void (*S)(double *, double *), Vecr q, Matr Ww, double dt, Vecr dX,
        int N, int V, int ndim, Matr DG_MAT, Matr DERVALS, Matr DG_DER,
        Vecr WGHTS) {

  int Nd = pow(N, ndim);
  MatMap qmat(q.data(), N * Nd, V, OuterStride(V));
  Mat tmp = rhs(F, B, S, qmat, Ww, dt, dX, N, V, DERVALS, WGHTS, DG_DER);

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
  VecMap ret(tmp.data(), N * Nd * V);
  return ret;
}

void initial_condition(Matr Ww, Matr w, Vecr WGHTS, Matr ENDVALS, int ndim) {

  int N = WGHTS.size();
  int Nd = pow(N, ndim);

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

Mat dg_predictor(void (*F)(double *, double *, int),
                 void (*B)(double *, double *, int),
                 void (*S)(double *, double *), Matr wh, double dt, Vecr dX,
                 bool STIFF, int N) {

  int ndim = dX.size();
  int Nd = pow(N, ndim);
  int V = wh.cols();

  Mat qh(wh.rows() * N, V);

  std::vector<poly> basis = basis_polys(N);
  Vec NODES = scaled_nodes(N);
  Vec WGHTS = scaled_weights(N);

  Mat DERVALS = derivative_values(basis, NODES);
  Mat ENDVALS = end_values(basis);

  Mat DG_END = end_value_products(basis);
  Mat DG_DER = derivative_products(basis, NODES, WGHTS);
  Mat DG_MAT = DG_END - DG_DER.transpose();

  std::vector<Mat> tmp(ndim + 1);
  tmp[0] = DG_MAT;
  for (int i = 0; i < ndim; i++)
    tmp[i + 1] = WGHTS.asDiagonal();

  Dec DG_U(kron(tmp));

  Mat Ww(N * Nd, V);
  Mat q0(N * Nd, V);

  for (int ind = 0; ind < wh.size(); ind += Nd * V) {

    MatMap wi(wh.data() + ind, Nd, V, OuterStride(V));
    MatMap qi(qh.data() + ind * N, N * Nd, V, OuterStride(V));

    initial_condition(Ww, wi, WGHTS, ENDVALS, ndim);

    using std::placeholders::_1;
    VecFunc obj_bound = std::bind(obj, F, B, S, _1, Ww, dt, dX, N, V, ndim,
                                  DG_MAT, DERVALS, DG_DER, WGHTS);

    initial_guess(q0, wi);

    if (STIFF) {
      VecMap q0v(q0.data(), N * Nd * V);
      qh.block(ind * N, 0, N * Nd, V) = nonlin_solve(obj_bound, q0v, DG_TOL);
    } else {

      for (int count = 0; count < DG_IT; count++) {

        Mat q1 = DG_U.solve(
            rhs(F, B, S, q0, Ww, dt, dX, N, V, DERVALS, WGHTS, DG_DER));

        aMat absDiff = (q1 - q0).array().abs();

        if ((absDiff > DG_TOL * (1. + q0.array().abs())).any()) {
          q0 = q1;
          continue;
        } else {
          qi = q1;
          break;
        }
      }
    }
  }
  return qh;
}
