#ifndef NEWTON_KRYLOV_H
#define NEWTON_KRYLOV_H

#include "../../types.h"

class KrylovJacobian {
  VecFunc func;
  int maxiter;
  int inner_m;
  unsigned int outer_k;

  Vec x0;
  Vec f0;
  double rdiff;
  double omega;
  std::vector<Vec> outer_v;
  Mat M;

public:
  void update_diff_step();
  KrylovJacobian(Vecr x, Vecr f, VecFunc F);
  Vec matvec(Vecr v);
  Vec psolve(Vecr v);
  Vec solve(Vecr rhs, double tol);
  void update(Vecr x, Vecr f);
};

class TerminationCondition {
  double f_tol;
  double f_rtol;
  double x_tol;
  double x_rtol;
  double f0_norm;

public:
  TerminationCondition(double ftol, double frtol, double xtol, double xrtol);
  int check(Vecr f, Vecr x, Vecr dx);
};

Vec nonlin_solve(VecFunc F, Vecr x, double f_tol = std::pow(mEPS, 1. / 3.),
                 double f_rtol = INF, double x_tol = INF, double x_rtol = INF);

#endif // NEWTON_KRYLOV_H