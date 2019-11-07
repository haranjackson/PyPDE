#include "ader.h"
#include <cstddef>

int main() {

  void (*F)(double *, double *, int) = NULL;
  void (*B)(double *, double *, int) = NULL;
  void (*S)(double *, double *) = NULL;

  int ndim = 2;
  int _nX[2] = {5, 5};
  double _u[50];
  double tf = 1.;
  double _dX[2] = {0.2, 0.2};
  double CFL = 0.9;
  int _boundaryTypes[2] = {0, 0};
  bool STIFF = true;
  int FLUX = 0;
  int N = 2;
  int V = 2;
  int ndt = 100;
  double _ret[5000];

  ader_solver(F, B, S, _u, tf, _nX, ndim, _dX, CFL, _boundaryTypes, STIFF, FLUX,
              N, V, ndt, _ret);
  return 0;
}