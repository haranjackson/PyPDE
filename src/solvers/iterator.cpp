#include "../etc/grid.h"
#include "../etc/system.h"
#include "../etc/types.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "weno/weno.h"

double timestep(void (*F)(double *, double *, int),
                void (*B)(double *, double *, int), Matr u, aVecr dX,
                double CFL, double t, double tf, int count) {

  int ndim = dX.size();

  double MAX = 0.;
  for (int ind = 0; ind < u.rows(); ind++) {

    double tmp = 0.;
    for (int d = 0; d < ndim; d++)
      tmp += max_abs_eigs(F, B, u.row(ind), d) / dX(d);

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

void iterator(void (*F)(double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr u, double tf, iVecr nX, aVecr dX, double CFL,
              iVecr boundaryTypes, bool STIFF, int FLUX, int N, int ndt,
              Matr ret) {

  int V = u.size() / nX.prod();
  Mat uprev = u;

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  while (t < tf) {

    double dt = timestep(F, B, u, dX, CFL, t, tf, count);

    Mat ub = boundaries(u, nX, boundaryTypes, N);

    Mat wh = weno_reconstruction(ub, nX, N, V);

    Mat qh = dg_predictor(F, B, S, wh, dt, dX, STIFF, N);

    finite_volume(F, B, S, u, qh, nX, dt, dX, FLUX, N);

    t += dt;
    count += 1;

    if (t >= double(pushCount + 1) / double(ndt) * tf) {
      ret.row(pushCount) = VecMap(u.data(), u.size());
      pushCount += 1;
    }

    if (u.array().isNaN().any()) {
      ret.row(pushCount) = VecMap(uprev.data(), uprev.size());
      ret.row(pushCount + 1) = VecMap(u.data(), u.size());
    }

    uprev = u;
  }

  ret.row(ndt - 1) = VecMap(u.data(), u.size());
}
