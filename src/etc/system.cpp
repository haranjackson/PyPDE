#include "NumericalDiff.h"
#include "types.h"

Mat system_matrix(void (*F)(double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, int d) {

  Mat dF = df(F, q, d, Forward);

  int n = q.size();
  Mat b(n, n);
  B(b.data(), q.data(), d);

  return dF + b;
}

double max_abs_eigs(void (*F)(double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, int d) {

  Mat jac = system_matrix(F, B, q, d);
  Eigen::EigenSolver<Mat> es(jac);
  return es.eigenvalues().array().abs().maxCoeff()
}
