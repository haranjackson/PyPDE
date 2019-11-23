#ifndef FV_H
#define FV_H

#include "../../grid/indexing.h"
#include "../../types.h"

class Indexer {
private:
  iVec nX;
  iVec nX2;

  bool in_bounds(iVecr inds, iVecr bounds, int offset) {
    for (int i = 0; i < inds.size(); i++) {
      if (inds(i) + offset < 0 || inds(i) + offset >= bounds(i))
        return false;
    }
    return true;
  }

public:
  int u;
  int qh;
  bool valid;

  Indexer(iVecr _nX, iVecr _nX2) : nX(_nX), nX2(_nX2) {}

  void set_indices(iVecr inds) {
    u = index(inds, nX, -1);
    qh = index(inds, nX2, 0);
    valid = in_bounds(inds, nX, -1);
  }
};

class FVSolver {
private:
  void (*F)(double *, double *, double *, int);
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

  bool secondOrder;

  Vec NODES;
  Vec WGHTS;

  Mat DERVALS;
  Mat ENDVALS;

  void centers(Matr u, Matr qh, double dt);

  void qh_derivatives(Matr dqh, Matr qh);

  void calculate_endpoints(Matr dqh, Matr qh0, Matr qh1, Matr q0, Matr q1,
                           Matr dq0, Matr dq1, int ind0, int ind1, int d);

  void interfaces(Matr u, Matr qh, double dt);

public:
  FVSolver(void (*_F)(double *, double *, double *, int),
           void (*_B)(double *, double *, int), void (*_S)(double *, double *),
           iVecr _nX, Vecr _dX, int _FLUX, int _N, int _V, bool _secondOrder);

  void apply(Matr u, Matr qh, double dt);
};

#endif // FV_H
