#include "NumericalDiff.h"
#include "eigen3/Eigenvalues"
#include "types.h"

#include <iostream>

Mat system_matrix(void (*F)(double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, int d) {

  int V = q.size();

  Mat M(V, V);

  if (F == NULL)
    M.setZero();
  else
    df(M, F, q, d, Forward);

  if (B != NULL) {

    Mat b(V, V);
    B(b.data(), q.data(), d);
    M += b;
  }

  return M;
}

double max_abs_eigs(void (*F)(double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, int d) {

  Mat jac = system_matrix(F, B, q, d);

  Eigen::EigenSolver<Mat> es(jac);
  return es.eigenvalues().array().abs().maxCoeff();
}
