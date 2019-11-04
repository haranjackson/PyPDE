#include "../../etc/indexing.h"
#include "../../etc/types.h"
#include "../poly/basis.h"
#include "../poly/evaluations.h"
#include "fluxes.h"

void centers(void (*B)(double *, double *, int), void (*S)(double *, double *),
             Vecr u, Matr qh, iVecr nX, double dt, Vecr dX, Vecr WGHTS,
             Matr DERVALS) {

  int ndim = nX.size();
  int N = WGHTS.size();
  int V = u.size() / nX.prod();

  int Nd = pow(N, ndim);

  iVec nX2 = nX;
  nX2.array() += 2;

  Mat dqh_dx(Nd, V);
  std::vector<Mat> dq(ndim);
  for (int d = 0; d < ndim; d++)
    dq[d] = dqh_dx;

  Mat b(V, V);
  Vec s(V);

  iVec indsOuter = iVec::Zero(ndim);
  iVec indsInner = iVec::Zero(ndim);

  int uCell = 0;
  while (uCell < nX.prod()) {

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
          tmp *= WGHTS(indsInner[i]);

        u.segment(uCell * V, V) += tmp * s;

        update_inds(indsInner, N);
        idx++;
      }
    }

    update_inds(indsOuter, nX);
    uCell++;
  }
}

void interfs(void (*F)(double *, double *, int),
             void (*B)(double *, double *, int), Vecr u, Matr qh, iVecr nX,
             double dt, Vecr dX, int FLUX, Vecr NODES, Vecr WGHTS,
             Matr ENDVALS) {

  int ndim = nX.size();
  int N = NODES.size();
  int V = u.size() / nX.prod();

  int Nd = pow(N, ndim);
  int Nd_ = pow(N, ndim - 1);

  Vec f(V);
  Vec b = Vec::Zero(V);

  Mat q0(Nd_, V);
  Mat q1(Nd_, V);

  iVec nX1 = nX;
  iVec nX2 = nX;
  nX1.array() += 1;
  nX2.array() += 2;

  iVec indsOuter = iVec::Zero(ndim);
  iVec indsInner = iVec::Zero(ndim - 1);

  int uCell = 0;
  while (uCell < nX1.prod()) { // (i = 0 ... (nX+1); j = 0 ... (nY+1); etc

    int uind0 = index(indsOuter, nX, -1) * V; // (i-1, j-2, ...)

    for (int t = 0; t < WGHTS.size(); t++) {

      int ind0 = (index(indsOuter, nX2, 0) * N + t) * Nd * V;
      MatMap qh0(qh.data() + ind0, Nd, V, OuterStride(V));

      for (int d = 0; d < ndim; d++) {

        int ind = indsOuter(d);
        double c = dt * WGHTS(t) / (2. * dX(d));

        indsOuter(d) += 1;
        int uind1 = index(indsOuter, nX, -1) * V;
        int ind1 = (index(indsOuter, nX2, 0) * N + t) * Nd * V;
        indsOuter(d) -= 1;

        MatMap qh1(qh.data() + ind1, Nd, V, OuterStride(V));

        endpts(q0, qh0, d, 1, ENDVALS, ndim);
        endpts(q1, qh1, d, 0, ENDVALS, ndim);

        int idx = 0;
        while (idx < Nd_) {

          switch (FLUX) {
          case OSHER:
            f = D_OSH(F, B, q0.row(idx), q1.row(idx), d, NODES, WGHTS);
            break;
          case ROE:
            f = D_ROE(F, B, q0.row(idx), q1.row(idx), d, NODES, WGHTS);
            break;
          case RUSANOV:
            f = D_RUS(F, B, q0.row(idx), q1.row(idx), d);
            break;
          }

          if (B != NULL)
            b = Bint(B, q0.row(idx), q1.row(idx), 0, NODES, WGHTS);

          double tmp = c;
          for (int i = 0; i < ndim - 1; i++)
            tmp *= WGHTS(indsInner[i]);

          if (ind > 0)
            u.segment(uind0, V) -= tmp * (b + f);
          if (ind < nX(d))
            u.segment(uind1, V) -= tmp * (b - f);

          update_inds(indsInner, N);
          idx++;
        }
      }
    }

    update_inds(indsOuter, nX1);
    uCell++;
  }
}

void fv_launcher(void (*F)(double *, double *, int),
                 void (*B)(double *, double *, int),
                 void (*S)(double *, double *), Vecr u, Matr qh, iVecr nX,
                 double dt, Vecr dX, int FLUX, int N) {

  Vec NODES = scaled_nodes(N);
  Vec WGHTS = scaled_weights(N);
  std::vector<poly> basis = basis_polys(N);

  Mat ENDVALS = end_values(basis);
  Mat DERVALS = derivative_values(basis, NODES);

  if (B != NULL || S != NULL)
    centers(B, S, u, qh, nX, dt, dX, WGHTS, DERVALS);

  if (F != NULL || B != NULL)
    interfs(F, B, u, qh, nX, dt, dX, FLUX, NODES, WGHTS, ENDVALS);
}
