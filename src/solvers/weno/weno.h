#ifndef WENO_H
#define WENO_H

#include "../../types.h"

class WenoSolver {
private:
  const double LAMS = 1.;   // WENO side stencil weighting
  const double LAMC = 1e5;  // WENO central stencil weighting
  const double EPS = 1e-14; // WENO epsilon parameter
  const double r = 8.;      // WENO exponent parameter

  iVec nX;
  int N;
  int V;

  int FN2;
  int CN2;

  Dec ML;
  Dec MR;
  Dec MCL;
  Dec MCR;
  Mat SIG;

  void coeffs_inner(Matr w, Dec M, Matr dataBlock, Matr num, Vecr den,
                    double LAM);

  Vec coeffs(Matr data);

public:
  WenoSolver(iVecr _nX, int _N, int _V);

  Mat reconstruction(Matr ub);
};

#endif // WENO_H
