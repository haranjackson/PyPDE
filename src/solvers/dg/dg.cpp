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

Mat rhs(Matr q, Matr Ww, double dt, double dx, double dy, int N, int V,
        Matr DERVALS, Vecr WGHTS, Matr DG_DER) {

  Mat ret(N * N * N, V);
  Mat dq_dx(N * N * N, V);
  Mat dq_dy(N * N * N, V);

  Mat f = Mat::Zero(N * N * N, V);
  Mat g = Mat::Zero(N * N * N, V);

  Vec tmpx(V);
  Vec tmpy(V);

  for (int t = 0; t < N; t++) {
    derivs(dq_dx.block(t * N * N, 0, N * N, V), q.block(t * N * N, 0, N * N, V),
           0, DERVALS);
    derivs(dq_dy.block(t * N * N, 0, N * N, V), q.block(t * N * N, 0, N * N, V),
           1, DERVALS);
  }
  dq_dx /= dx;
  dq_dy /= dy;

  int ind = 0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        source(ret.row(ind), q.row(ind));
        Bdot(tmpx, q.row(ind), dq_dx.row(ind), 0);
        Bdot(tmpy, q.row(ind), dq_dy.row(ind), 1);
        ret.row(ind) -= tmpx + tmpy;
        ret.row(ind) *= WGHTS(t) * WGHTS(i) * WGHTS(j);
        flux(f.row(ind), q.row(ind), 0);
        flux(g.row(ind), q.row(ind), 1);
        ind += 1;
      }

  Mat f2(N * N * N, V);
  Mat g2(N * N * N, V);
  for (int i = 0; i < N * N * N; i++) {
    f2.row(i) = WGHTS(i % N) * f.row(i);
  }
  for (int i = 0; i < N * N; i++) {
    g2.block(i * N, 0, N, V) = DG_DER * g.block(i * N, 0, N, V);
  }

  ind = 0;
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int indi = t * N + i;
        int indj = t * N + j;
        ret.block(indi * N, 0, N, V) -=
            WGHTS(t) * DG_DER(i, j) * f2.block(indj * N, 0, N, V) / dx;
      }
      ret.block(ind * N, 0, N, V) -=
          WGHTS(t) * WGHTS(i) * g2.block(ind * N, 0, N, V) / dy;
      ind += 1;
    }

  ret *= dt;
  ret += Ww;
  return ret;
}

Vec obj(Vecr q, Matr Ww, double dt, double dx, double dy, int N, int V,
        Matr DG_MAT, Matr DERVALS, Matr DG_DER, Vecr WGHTS) {

  MatMap qmat(q.data(), OuterStride(V));
  Mat tmp = rhs(qmat, Ww, dt, dx, dy, N, V, DERVALS, WGHTS, DG_DER);

  for (int t = 0; t < N; t++)
    for (int k = 0; k < N; k++)
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          int indt = (t * N + i) * N + j;
          int indk = (k * N + i) * N + j;
          tmp.row(indt) -=
              (DG_MAT(t, k) * WGHTS(i) * WGHTS(j)) * qmat.row(indk);
        }
  VecMap ret(tmp.data(), N * N * N * V);
  return ret;
}

void initial_condition(Matr Ww, Matr w, Vecr WGHTS, Matr ENDVALS) {

  int N = WGHTS.size();
  for (int t = 0; t < N; t++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        Ww.row(t * N * N + i * N + j) =
            ENDVALS(0, t) * WGHTS(i) * WGHTS(j) * w.row(i * N + j);
}

void predictor(Vecr qh, Vecr wh, double dt, Vecr dX, int ncell, bool STIFF,
               int N, int V) {

  int ndim = dX.size();

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

  Mat Ww(N * N * N, V);
  Mat q0(N * N * N, V);

  double dx = dX(0);
  double dy = dX(1);

  for (int ind = 0; ind < ncell; ind++) {

    MatMap wi(wh.data() + (ind * N * N * V), N * N, V, OuterStride(V));
    MatMap qi(qh.data() + (ind * N * N * N * V), N * N * N, V, OuterStride(V));

    initial_condition(Ww, wi, WGHTS, ENDVALS);

    using std::placeholders::_1;
    VecFunc obj_bound = std::bind(obj, _1, Ww, dt, dx, dy, N, V, DG_MAT,
                                  DERVALS, DG_DER, WGHTS);

    initial_guess(q0, wi);

    if (STIFF) {
      VecMap q0v(q0.data(), N * N * N * V);
      qh.segment(ind * N * N * N * V, N * N * N * V) =
          nonlin_solve(obj_bound, q0v, DG_TOL);
    } else {

      Mat q1(N * N * N, V);
      aMat absDiff(N * N * N, V);

      for (int count = 0; count < DG_IT; count++) {

        q1 = DG_U.solve(rhs(q0, Ww, dt, dx, dy, N, V, DERVALS, WGHTS, DG_DER));

        absDiff = (q1 - q0).array().abs();

        if ((absDiff > DG_TOL * (1 + q0.array().abs())).any()) {
          q0 = q1;
          continue;
        } else {
          qi = q1;
          break;
        }
      }
    }
  }
}
