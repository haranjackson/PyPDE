#include <vector>

#include "../../scipy/math/polynomials.h"
#include "../../types.h"

int ceil(int x, int y) { return x / y + (x % y != 0); }

std::vector<Mat> coefficient_matrices(const std::vector<poly> &basis, int FN2,
                                      int CN2) {
  // Generate linear systems governing  coefficients of the basis polynomials

  int N = basis.size();

  Mat mL(N, N);
  Mat mR(N, N);
  Mat mCL(N, N);
  Mat mCR(N, N);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      poly Pj = basis[j].intt();
      mL(i, j) = Pj.eval(i - N + 2) - Pj.eval(i - N + 1);
      mR(i, j) = Pj.eval(i + 1) - Pj.eval(i);
      mCL(i, j) = Pj.eval(i - CN2 + 1) - Pj.eval(i - CN2);
      mCR(i, j) = Pj.eval(i - FN2 + 1) - Pj.eval(i - FN2);
    }

  std::vector<Mat> ret(4);
  ret[0] = mL;
  ret[1] = mR;
  ret[2] = mCL;
  ret[3] = mCR;
  return ret;
}

Mat oscillation_indicator(const std::vector<poly> &basis) {
  // Generate the oscillation indicator matrix from a set of basis polynomials

  int N = basis.size();
  Mat ret = Mat::Zero(N, N);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int a = 1; a < N; a++) {
        poly p = basis[i].diff(a) * basis[j].diff(a);
        poly P = p.intt();
        ret(i, j) += P.eval(1) - P.eval(0);
      }

  return ret;
}
