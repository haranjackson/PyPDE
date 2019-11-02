#include "NumericalDiff.h"
#include "eigen3/Eigenvalues"
#include "types.h"

Mat system_matrix(void (*F)(double *, double *, int),
                  void (*B)(double *, double *, int), Vecr q, int d) {

  int V = q.size();

  Mat dF;

  if (F == NULL)
    dF = Mat::Zero(V, V);
  else
    dF = df(F, q, d, Forward);

  if (B != NULL) {

    Mat b(V, V);
    B(b.data(), q.data(), d);
    dF += b;
  }

  return dF;
}

double max_abs_eigs(void (*F)(double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, int d) {

  Mat jac = system_matrix(F, B, q, d);
  Eigen::EigenSolver<Mat> es(jac);
  return es.eigenvalues().array().abs().maxCoeff();
}
