#include "CustomMatProd.h"
#include "NumericalDiff.h"
#include "eigen3/Eigenvalues"
#include "types.h"

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

double max_abs_eig_spectra(Matr M) {

  int n = M.rows();

  CustomMatProd op(M);
  SpectraEigSolver eigs(&op, 1, n);
  eigs.init();
  eigs.compute();

  return std::abs(eigs.eigenvalues()(0));
}

double max_abs_eigs(void (*F)(double *, double *, int),
                    void (*B)(double *, double *, int), Vecr q, int d) {

  Mat jac = system_matrix(F, B, q, d);

  // TODO: work out when it is optimal to use Spectra
  // (presumably for larger matrices)
  if (jac.cols() > 5)
    return max_abs_eig_spectra(jac);
  else
    return jac.eigenvalues().array().abs().maxCoeff();
}
