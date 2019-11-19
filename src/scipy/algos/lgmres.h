#ifndef LGMRES_H
#define LGMRES_H

#include "../../types.h"
#include "newton_krylov.h"

class System {
  Mat A;
  Mat M;

public:
  Vec call(Vecr x) { return A * x; }
  Vec prec(Vecr x) { return M * x; }
  void set_A(Mat A_) { A = A_; }
  void set_M(Mat M_) { M = M_; }
};

Vec lgmres(VecFunc matvec, VecFunc psolve, Vecr b, Vec x,
           std::vector<Vec> &outer_v, const double tol, const int maxiter,
           const int inner_m, const int outer_k);

Vec lgmres_wrapper(Matr A, Vecr b, Vecr x0, Matr M, const double tol,
                   const int maxiter, const int inner_m, const int outer_k,
                   std::vector<Vec> &outer_v);

#endif // LGMRES_H
