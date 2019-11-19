#ifndef FV_H
#define FV_H

#include "../../types.h"

class FVSolver {
private:
  void (*F)(double *, double *, int);
  void (*B)(double *, double *, int);
  void (*S)(double *, double *);

  iVec nX;
  Vec dX;
  int ndim;

  iVec nX1;
  iVec nX2;

  int FLUX;

  int N;
  int V;
  int Nd;
  int Nd_;

  Vec NODES;
  Vec WGHTS;

  Mat DERVALS;
  Mat ENDVALS;

  void centers(Matr u, Matr qh, double dt);

  void interfaces(Matr u, Matr qh, double dt);

public:
  FVSolver(void (*_F)(double *, double *, int),
           void (*_B)(double *, double *, int), void (*_S)(double *, double *),
           iVecr _nX, Vecr _dX, int _FLUX, int _N, int _V);

  void apply(Matr u, Matr qh, double dt);
};

#endif // FV_H
