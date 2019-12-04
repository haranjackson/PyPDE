#include "../src/api.h"
#include <cstddef>

void F(double *ret, double *Q, double *dQ, int d) {

  double r = Q[0];
  double ru = Q[1];
  double rv = Q[2];
  double rE = Q[3];

  double u = ru / r;
  double v = rv / r;
  double E = rE / r;

  double g = 1.4;
  double p = E * r * (g - 1);

  if (d == 0) {
    ret[0] = ru;
    ret[1] = ru * u + p;
    ret[2] = rv * u;
    ret[3] = rE * u + p * u;
  } else {
    ret[0] = rv;
    ret[1] = ru * v;
    ret[2] = rv * v + p;
    ret[3] = rE * v + p * v;
  }
}

void fill_u(double *_u, int nX, int nY, int V) {
  for (int i = 0; i < nX; i++)
    for (int j = 0; j < nY; j++) {
      int ind = (i * nY + j) * V;
      _u[ind + 0] = 1.;
      _u[ind + 1] = 2.;
      _u[ind + 2] = 3.;
      _u[ind + 3] = 4.;
    }
}

int main() {

  void (*B)(double *, double *, int) = NULL;
  void (*S)(double *, double *) = NULL;

  int nX = 30;
  int nY = 10;

  int _nX[2] = {nX, nY};
  int ndim = 2;
  int V = 4;

  double _u[nX * nY * V];
  double _dX[2] = {1. / double(nX), 1. / double(nY)};

  fill_u(_u, nX, nY, V);

  double tf = 1.;
  double CFL = 0.9;
  int _boundaryTypes[2] = {0, 0};

  bool STIFF = false;
  int FLUX = 0;
  int N = 2;

  int ndt = 100;
  double _ret[ndt * nX * nY * V];

  pde_solver(F, B, S, true, false, false, _u, tf, _nX, ndim, _dX, CFL,
             _boundaryTypes, STIFF, FLUX, N, V, ndt, false, _ret, -1);
  return 0;
}