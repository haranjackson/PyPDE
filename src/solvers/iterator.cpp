#include <iostream>

#include "../etc/grid.h"
#include "../etc/system.h"
#include "../etc/types.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "weno/weno.h"

void stepper(void (*F)(double *, double *, int),
             void (*B)(double *, double *, int), void (*S)(double *, double *),
             Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX, bool STIFF,
             int FLUX, int N, int V) {

  Mat wh = weno_launcher(ub, nX, N, V);

  Mat qh = predictor(F, B, S, wh, dt, dX, STIFF, N, V);

  fv_launcher(F, B, S, u, qh, nX, dt, dX, FLUX, N);
}

double timestep(void (*F)(double *, double *, int),
                void (*B)(double *, double *, int), Vecr u, aVecr dX,
                double CFL, double t, double tf, int count, int V) {

  double MAX = 0.;

  int ndim = dX.size();

  for (int ind = 0; ind < u.size(); ind += V) {

    double tmp = 0.;
    for (int d = 0; d < ndim; d++)
      tmp += max_abs_eigs(F, B, u.segment(ind, V), d) / dX(d);

    MAX = std::max(MAX, tmp);
  }

  double dt = CFL / MAX;

  if (count <= 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;
  else
    return dt;
}

std::vector<Vec> iterator(void (*F)(double *, double *, int),
                          void (*B)(double *, double *, int),
                          void (*S)(double *, double *), Vecr u, double tf,
                          iVecr nX, aVecr dX, double CFL, iVecr boundaryTypes,
                          bool STIFF, int FLUX, int N, int ndt) {

  int V = u.size() / nX.prod();
  Vec uprev(u.size());
  std::vector<Vec> ret(ndt);

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  double dt = 0.;

  while (t < tf) {

    uprev = u;

    dt = timestep(F, B, u, dX, CFL, t, tf, count, V);

    Vec ub = boundaries(u, nX, boundaryTypes, N);

    stepper(F, B, S, u, ub, nX, dt, dX, STIFF, FLUX, N, V);

    t += dt;
    count += 1;

    if (t >= double(pushCount + 1) / double(ndt) * tf) {
      ret[pushCount] = u;
      pushCount += 1;
    }

    if (u.array().isNaN().any()) {
      ret[pushCount] = uprev;
      ret[pushCount + 1] = u;
      return ret;
    }
  }

  ret[ndt - 1] = u;
  return ret;
}
