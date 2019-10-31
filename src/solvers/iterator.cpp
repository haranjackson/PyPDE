#include <iostream>

#include "../etc/grid.h"
#include "../etc/types.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "steppers.h"
#include "weno/weno.h"

void stepper(Vecr u, Vecr ub, iVecr nX, double dt, Vecr dX, bool STIFF,
             int FLUX, int N, int V) {

  int ndim = nX.size();
  Vec wh(extended_dimensions(nX, 1) * int(pow(N, ndim)) * V);
  Vec qh(extended_dimensions(nX, 1) * int(pow(N, ndim + 1)) * V);

  weno_launcher(wh, ub, nX);

  predictor(qh, wh, dt, dX, STIFF);

  fv_launcher(u, qh, nX, dt, dX, FLUX);
}

double timestep(Vecr u, aVecr dX, double CFL, double t, double tf, int count,
                int V) {

  double MAX = 0.;

  int ndim = dX.size();
  int ncell = u.size() / V;

  for (int ind = 0; ind < ncell; ind++) {
    double tmp = 0.;
    for (int d = 0; d < ndim; d++)
      tmp += max_abs_eigs(u.segment(ind * V, V), d) / dX(d);
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

std::vector<Vec> iterator(Vecr u, double tf, iVecr nX, aVecr dX, double CFL,
                          iVecr boundaryTypes, bool STIFF, int FLUX, int N,
                          int ndt) {

  int V = u.size() / nX.prod();
  Vec uprev(u.size());
  std::vector<Vec> ret(ndt);

  Vec ub(extended_dimensions(nX, N) * V);

  int ncell = u.size() / V;

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  double dt = 0.;

  while (t < tf) {

    uprev = u;

    dt = timestep(u, dX, CFL, t, tf, count, V);

    boundaries(u, ub, nX, boundaryTypes, N, V);

    stepper(u, ub, nX, dt, dX, STIFF, FLUX);

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
