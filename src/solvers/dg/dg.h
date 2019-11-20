#ifndef DG_H
#define DG_H

#include "../../types.h"

class DGSolver {
private:
  const int DG_IT = 50; // No. of iterations of non-Newton solver attempted
  const double DG_TOL = 6e-6; // Convergence tolerance

  void (*F)(double *, double *, double *, int);
  void (*B)(double *, double *, int);
  void (*S)(double *, double *);

  Vec dX;
  int ndim;

  bool STIFF;

  int N;
  int V;
  int Nd;

  Vec NODES;
  Vec WGHTS;

  Mat DERVALS;
  Mat ENDVALS;

  Mat DG_MAT;
  Dec DG_U;

  Mat rhs(Matr q, Matr Ww, double dt);

  Vec obj(Vecr q, Matr Ww, double dt);

  void initial_condition(Matr Ww, Matr w);

  Mat stiff_solve(Matr q0, Matr Ww, double dt);

  Mat nonstiff_solve(Matr q0, Matr Ww, double dt);

public:
  DGSolver(void (*_F)(double *, double *, double *, int),
           void (*_B)(double *, double *, int), void (*_S)(double *, double *),
           Vecr _dX, bool _STIFF, int _N, int _V);

  Mat predictor(Matr wh, double dt);
};

#endif // DG_H
