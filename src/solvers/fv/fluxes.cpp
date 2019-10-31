#include <cmath>

#include "eigen3/Eigenvalues"

#include "../../etc/system.h"
#include "../../etc/types.h"

Vec Bint(void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
         Vecr NODES, Vecr WGHTS) {
  // Returns the jump matrix for B, in the dth direction

  Vec Δq = qR - qL;
  int V = Δq.size();

  Vec q(V);
  Mat b(V, V);
  Mat ret = Mat::Zero(V, V);

  for (int i = 0; i < NODES.size(); i++) {
    q = qL + NODES(i) * Δq;
    B(b.data(), q.data(), d);
    ret += WGHTS(i) * b;
  }
  return ret * Δq;
}

Vec D_OSH(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
          Vecr NODES, Vecr WGHTS) {
  // Returns the Osher flux component, in the dth direction

  Vec Δq = qL - qR;
  cVec Δqc = cVec(Δq);
  int V = Δq.size();

  Vec q(V);
  cVec b(V);
  Mat J(V, V);
  cVec ret = cVec::Zero(V);
  Eigen::EigenSolver<Mat> es;

  for (int i = 0; i < NODES.size(); i++) {

    q = qR + NODES(i) * Δq;
    J = system_matrix(F, B, q, d);
    es.compute(J);

    b = es.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
        es.eigenvalues().array().abs();
    ret += WGHTS(i) * (es.eigenvectors() * b);
  }
  Vec f = ret.real();
  F(f.data(), qL.data(), d);
  F(f.data(), qR.data(), d);
  return f;
}

Vec D_ROE(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
          Vecr NODES, Vecr WGHTS) {
  // Returns the Osher flux component, in the dth direction

  Vec Δq = qL - qR;
  cVec Δqc = cVec(Δq);
  int V = Δq.size();

  Vec q(V);
  Mat J = Mat::Zero(V, V);

  for (int i = 0; i < NODES.size(); i++) {
    q = qR + NODES(i) * Δq;
    J += WGHTS(i) * system_matrix(F, B, q, d);
  }
  Eigen::EigenSolver<Mat> es(J);
  cVec b = es.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
           es.eigenvalues().array().abs();

  Vec f = (es.eigenvectors() * b).real();
  F(f.data(), qL.data(), d);
  F(f.data(), qR.data(), d);
  return f;
}

Vec D_RUS(void (*F)(double *, double *, int),
          void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d) {

  double max1 = max_abs_eigs(F, B, qL, d);
  double max2 = max_abs_eigs(F, B, qR, d);

  Vec f = std::max(max1, max2) * (qL - qR);
  F(f.data(), qL.data(), d);
  F(f.data(), qR.data(), d);
  return f;
}
