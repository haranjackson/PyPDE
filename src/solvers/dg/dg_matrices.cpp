#include "../../scipy/math/polynomials.h"
#include "../../types.h"

Mat kron(std::vector<Mat> &mats) {
  int idx = mats.size();
  int m0 = mats[idx - 2].rows();
  int m1 = mats[idx - 1].rows();
  int n0 = mats[idx - 2].cols();
  int n1 = mats[idx - 1].cols();
  Mat ret(m0 * m1, n0 * n1);

  for (int i0 = 0; i0 < m0; i0++)
    for (int i1 = 0; i1 < m1; i1++) {
      int i = i0 * m1 + i1;
      for (int j0 = 0; j0 < n0; j0++)
        for (int j1 = 0; j1 < n1; j1++) {
          int j = j0 * n1 + j1;
          ret(i, j) = mats[idx - 2](i0, j0) * mats[idx - 1](i1, j1);
        }
    }
  if (mats.size() == 2) {
    return ret;
  } else {
    mats.pop_back();
    mats[idx - 2] = ret;
    return kron(mats);
  }
}

Mat end_value_products(const std::vector<poly> &basis) {
  // ret[i,j] = ψ_i(1) * ψ_j(1)
  int N = basis.size();
  Mat ret(N, N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      ret(i, j) = basis[i].eval(1) * basis[j].eval(1);
  return ret;
}

Mat derivative_products(const std::vector<poly> &basis, Vecr NODES,
                        Vecr WGHTS) {
  // ret[i,j] = < ψ_i , ψ_j' >
  int N = basis.size();
  Mat ret(N, N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      if (i == j) {
        double eval0 = basis[i].eval(0);
        double eval1 = basis[i].eval(1);
        ret(i, j) = (eval1 * eval1 - eval0 * eval0) / 2;
      } else
        ret(i, j) = WGHTS(i) * basis[j].diff(1).eval(NODES(i));
    }
  return ret;
}
