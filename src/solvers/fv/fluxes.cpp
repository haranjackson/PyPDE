#include "fluxes.h"
#include "../../eigs/system.h"
#include "../../types.h"
#include "eigen3/Eigenvalues"
#include <cmath>

FluxGenerator::FluxGenerator(void (*_F)(double *, double *, double *, int),
                             void (*_B)(double *, double *, int), Vecr _NODES,
                             Vecr _WGHTS, Vecr _dX, int _V, int _FLUX, int ndim)
    : F(_F), B(_B), NODES(_NODES), WGHTS(_WGHTS), dX(_dX), V(_V), FLUX(_FLUX) {

  N = NODES.size();
  fL = Vec(V);
  fR = Vec(V);
  q = Vec(V);
  dq = Mat(ndim, V);
  M = Mat(V, V);
}

Vec FluxGenerator::D_OSH(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Dq = qL - qR;
  Mat Ddq = dqL - dqR;

  cVec Dqc = cVec(Dq);

  cVec b(V);
  cVec ret = cVec::Zero(V);

  for (int i = 0; i < N; i++) {

    q = qR + NODES(i) * Dq;
    dq = dqR + NODES(i) * Ddq;

    M = system_matrix(F, B, q, dq, d);
    ES.compute(M);

    b = ES.eigenvectors().colPivHouseholderQr().solve(Dqc).array() *
        ES.eigenvalues().array().abs();
    ret += WGHTS(i) * (ES.eigenvectors() * b);
  }

  return ret.real();
}

Vec FluxGenerator::D_ROE(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Dq = qL - qR;
  Mat Ddq = dqL - dqR;

  cVec Dqc = cVec(Dq);

  M.setZero();

  for (int i = 0; i < N; i++) {

    q = qR + NODES(i) * Dq;
    dq = dqR + NODES(i) * Ddq;

    M += WGHTS(i) * system_matrix(F, B, q, dq, d);
  }
  ES.compute(M);

  cVec b = ES.eigenvectors().colPivHouseholderQr().solve(Dqc).array() *
           ES.eigenvalues().array().abs();

  return (ES.eigenvectors() * b).real();
}

Vec FluxGenerator::D_RUS(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d) {

  double max1 = max_abs_eigs(F, B, qL, dqL, d);
  double max2 = max_abs_eigs(F, B, qR, dqR, d);

  return std::max(max1, max2) * (qL - qR);
}

void FluxGenerator::flux(Vecr ret, Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d,
                         bool secondOrder) {

  if (FLUX == RUSANOV)
    ret = D_RUS(qL, qR, dqL, dqR, d);

  if (FLUX == ROE)
    ret = D_ROE(qL, qR, dqL, dqR, d);

  if (FLUX == OSHER)
    ret = D_OSH(qL, qR, dqL, dqR, d);

  F(fL.data(), qL.data(), dqL.data(), d);
  F(fR.data(), qR.data(), dqR.data(), d);
  ret += fL + fR;

  if (secondOrder) {
    double max1 = max_abs_eigs_second_order(F, qL, dqL, d, N, dX);
    double max2 = max_abs_eigs_second_order(F, qR, dqR, d, N, dX);
    ret += std::max(max1, max2) * (qL - qR);
  }
}

void FluxGenerator::Bint(Vecr ret, Vecr qL, Vecr qR, int d) {
  // Returns the jump matrix for B, in the dth direction

  Vec Dq = qR - qL;

  Mat b(V, V);
  M.setZero();

  for (int i = 0; i < N; i++) {
    q = qL + NODES(i) * Dq;
    B(b.data(), q.data(), d);
    M += WGHTS(i) * b;
  }
  ret = M * Dq;
}