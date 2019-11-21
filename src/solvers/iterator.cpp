#include "../grid/boundaries.h"
#include "../types.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "stepper.h"
#include "weno/weno.h"

#include <iostream>

void iterator(void (*F)(double *, double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr u, double tf, iVecr nX, aVecr dX, double CFL,
              iVecr boundaryTypes, bool STIFF, int FLUX, int N, int ndt,
              bool secondOrder, Matr ret) {

  int V = u.size() / nX.prod();
  Mat uprev = u;

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  iVec nXb = nX;
  nXb.array() += 2 * N;

  TimeStepper timeStepper(F, B, dX, N, V, CFL, tf, secondOrder);
  WenoSolver wenoSolver(nXb, N, V);
  DGSolver dgSolver(F, B, S, dX, STIFF, N, V);
  FVSolver fvSolver(F, B, S, nX, dX, FLUX, N, V, secondOrder);

  while (t < tf) {

    Mat ub = boundaries(u, nX, boundaryTypes, N);

    Mat wh = wenoSolver.reconstruction(ub);

    double dt = timeStepper.step(wh, t, count);

    Mat qh = dgSolver.predictor(wh, dt);

    fvSolver.apply(u, qh, dt);

    t += dt;
    count += 1;

    std::cout << "t = " << t << "\n";

    if (t >= double(pushCount + 1) / double(ndt) * tf) {
      ret.row(pushCount) = VecMap(u.data(), u.size());
      pushCount += 1;
    }

    if (u.array().isNaN().any()) {
      std::cout << "NaNs found";
      ret.row(pushCount) = VecMap(uprev.data(), uprev.size());
      ret.row(pushCount + 1) = VecMap(u.data(), u.size());
    }

    uprev = u;
  }

  ret.row(ndt - 1) = VecMap(u.data(), u.size());
}
