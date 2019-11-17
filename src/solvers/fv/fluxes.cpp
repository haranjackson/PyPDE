#include <cmath>

#include "eigen3/Eigenvalues"

#include "../../etc/system.h"
#include "../../etc/types.h"
#include "fluxes.h"

FluxGenerator::FluxGenerator(void (*_F)(double *, double *, int),
                             void (*_B)(double *, double *, int), Vecr _NODES,
                             Vecr _WGHTS, int _V, int _FLUX)
    : F(_F), B(_B), NODES(_NODES), WGHTS(_WGHTS), V(_V), FLUX(_FLUX) {

  N = NODES.size();
  fL = Vec(V);
  fR = Vec(V);
  q = Vec(V);
  M = Mat(V, V);
}

Vec FluxGenerator::D_OSH(Vecr qL, Vecr qR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Δq = qL - qR;
  cVec Δqc = cVec(Δq);

  cVec b(V);
  cVec ret = cVec::Zero(V);

  for (int i = 0; i < N; i++) {
    q = qR + NODES(i) * Δq;
    M = system_matrix(F, B, q, d);
    ES.compute(M);

    b = ES.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
        ES.eigenvalues().array().abs();
    ret += WGHTS(i) * (ES.eigenvectors() * b);
  }

  return ret.real();
}

Vec FluxGenerator::D_ROE(Vecr qL, Vecr qR, int d) {
  // Returns the Osher flux component, in the dth direction

  Vec Δq = qL - qR;
  cVec Δqc = cVec(Δq);

  M.setZero();

  for (int i = 0; i < N; i++) {
    q = qR + NODES(i) * Δq;
    M += WGHTS(i) * system_matrix(F, B, q, d);
  }
  ES.compute(M);
  cVec b = ES.eigenvectors().colPivHouseholderQr().solve(Δqc).array() *
           ES.eigenvalues().array().abs();

  return (ES.eigenvectors() * b).real();
}

Vec FluxGenerator::D_RUS(Vecr qL, Vecr qR, int d) {

  double max1 = max_abs_eigs(F, B, qL, d);
  double max2 = max_abs_eigs(F, B, qR, d);

  return std::max(max1, max2) * (qL - qR);
}

void FluxGenerator::flux(Vecr ret, Vecr qL, Vecr qR, int d) {

  F(fL.data(), qL.data(), d);
  F(fR.data(), qR.data(), d);

  switch (FLUX) {
  case RUSANOV:
    ret = D_RUS(qL, qR, d) + fL + fR;
  case ROE:
    ret = D_ROE(qL, qR, d) + fL + fR;
  case OSHER:
    ret = D_OSH(qL, qR, d) + fL + fR;
  default:
    break;
  }
}

void FluxGenerator::Bint(Vecr ret, Vecr qL, Vecr qR, int d) {
  // Returns the jump matrix for B, in the dth direction

  Vec Δq = qR - qL;

  Mat b(V, V);
  M.setZero();

  for (int i = 0; i < N; i++) {
    q = qL + NODES(i) * Δq;
    B(b.data(), q.data(), d);
    M += WGHTS(i) * b;
  }
  ret = M * Δq;
}