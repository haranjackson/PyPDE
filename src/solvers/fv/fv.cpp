#include "../../etc/types.h"
#include "../poly/basis.h"
#include "../poly/evaluations.h"
#include "fluxes.h"

int ind(int i, int t, int nt) { return i * nt + t; }

int ind(int i, int j, int t, int ny, int nt) { return (i * ny + j) * nt + t; }

void centers_inner(Vecr u, Vecr rec, iVecr nX, Vecr dX, int nt, int t,
                   double wght_t, Vecr WGHTS, int N, int V) {

  Mat dqh_dx(N * N, V);
  Mat dqh_dy(N * N, V);

  Vec qs(V);
  Vec dqdxs(V);
  Vec dqdys(V);
  Vec S(V);
  Vec tmpx(V);
  Vec tmpy(V);

  for (int i = 0; i < nX(0); i++)
    for (int j = 0; j < nX(1); j++) {

      int idx = ind(i + 1, j + 1, t, nX(1) + 2, nt) * N * N * V;

      MatMap qh(rec.data() + idx, OuterStride(V));
      derivs(dqh_dx, qh, 0);
      derivs(dqh_dy, qh, 1);

      for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++) {
          int s = a * N + b;
          qs = qh.row(s);
          dqdxs = dqh_dx.row(s);
          dqdys = dqh_dy.row(s);

          source(S, qs);

          Bdot(tmpx, qs, dqdxs, 0);
          Bdot(tmpy, qs, dqdys, 1);

          S -= tmpx / dx;
          S -= tmpy / dy;

          u.segment((i * nX(1) + j) * V, V) += wght_t * WGHTS(a) * WGHTS(b) * S;
        }
    }
}

void centers(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, Vecr WGHTS) {

  for (int t = 0; t < N; t++)
    centers_inner(u, rec, nX, dX, N, t, dt * WGHTS(t), WGHTS);
}

void interfs_inner(Vecr u, Vecr rec, iVecr nX, Vecr dX, int nt, int t,
                   double wghts_t, int FLUX, Vecr NODES, Vecr WGHTS,
                   Matr ENDVALS, int N, int V) {

  Mat q0(N, V);
  Mat q1(N, V);
  Vec f(V);
  Vec b(V);
  Vec u0(V);
  Vec u1(V);

  Vec xWGHTS = wghts_t / (2. * dX(0)) * WGHTS;
  Vec yWGHTS = wghts_t / (2. * dX(1)) * WGHTS;

  int NNV = N * N * V;

  for (int i = 0; i < nX(0) + 1; i++)
    for (int j = 0; j < nX(1) + 1; j++) {

      int uind0 = ind(i - 1, j - 1, nX(1)) * V;
      int ind0 = ind(i, j, t, nX(1) + 2, nt) * NNV;
      MatMap qh0(rec.data() + ind0, OuterStride(V));

      int uindx = ind(i, j - 1, nX(1)) * V;
      int indx = ind(i + 1, j, t, nX(1) + 2, nt) * NNV;

      MatMap qhx(rec.data() + indx, OuterStride(V));
      endpts(q0, qh0, 0, 1, ENDVALS);
      endpts(q1, qhx, 0, 0, ENDVALS);

      u0.setZero(V);
      u1.setZero(V);

      for (int s = 0; s < N; s++) {

        switch (FLUX) {
        case OSHER:
          f = D_OSH(q0.row(s), q1.row(s), 0, NODES, WGHTS);
          break;
        case ROE:
          f = D_ROE(q0.row(s), q1.row(s), 0, NODES, WGHTS);
          break;
        case RUSANOV:
          f = D_RUS(q0.row(s), q1.row(s), 0);
          break;
        }
        b = Bint(q0.row(s), q1.row(s), 0, NODES, WGHTS);

        if (i > 0)
          u0 += xWGHTS(s) * (b + f);
        if (i < nX(0))
          u1 += xWGHTS(s) * (b - f);
      }
      if (i > 0)
        u.segment(uind0, V) -= u0;
      if (i < nX(0))
        u.segment(uindx, V) -= u1;

      int uindy = ind(i - 1, j, nX(1)) * V;
      int indy = ind(i, j + 1, t, nX(1) + 2, nt) * NNV;

      MatMap qhy(rec.data() + indy, OuterStride(V));
      endpts(q0, qh0, 1, 1, ENDVALS);
      endpts(q1, qhy, 1, 0, ENDVALS);

      u0.setZero(V);
      u1.setZero(V);

      for (int s = 0; s < N; s++) {

        switch (FLUX) {
        case OSHER:
          f = D_OSH(q0.row(s), q1.row(s), 1, NODES, WGHTS);
          break;
        case ROE:
          f = D_ROE(q0.row(s), q1.row(s), 1, NODES, WGHTS);
          break;
        case RUSANOV:
          f = D_RUS(q0.row(s), q1.row(s), 1);
          break;
        }
        b = Bint(q0.row(s), q1.row(s), 1, NODES, WGHTS);

        if (j > 0)
          u0 += yWGHTS(s) * (b + f);
        if (j < nX(1))
          u1 += yWGHTS(s) * (b - f);
      }
      if (j > 0)
        u.segment(uind0, V) -= u0;
      if (j < nX(1))
        u.segment(uindy, V) -= u1;
    }
}

void interfs(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, int FLUX,
             Vecr NODES, Vecr WGHTS, Matr ENDVALS, int N, int V) {

  for (int t = 0; t < WGHTS.size(); t++)
    interfs_inner(u, rec, nX, dX, N, t, dt * WGHTS(t), FLUX, NODES, WGHTS,
                  ENDVALS, N, V);
}

void fv_launcher(Vecr u, Vecr rec, iVecr nX, double dt, Vecr dX, int FLUX) {

  Vec NODES = scaled_nodes(N);
  Vec WGHTS = scaled_weights(N);
  std::vector<poly> basis = basis_polys(N);
  Mat ENDVALS = end_values(basis);

  int ndim = nX.size();
  centers(u, rec, nX, dt, dX, WGHTS);
  interfs(u, rec, nX, dt, dX, FLUX, NODES, WGHTS, ENDVALS, N, V);
}
