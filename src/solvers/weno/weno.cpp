#include "weno.h"
#include "../../poly/basis.h"
#include "../../types.h"
#include "weno_matrices.h"
#include <cmath>

WenoSolver::WenoSolver(iVecr _nX, int _N, int _V) : nX(_nX), N(_N), V(_V) {

  FN2 = (int)floor((N - 1) / 2.);
  CN2 = (int)ceil((N - 1) / 2.);

  std::vector<poly> basis = basis_polys(N);
  std::vector<Mat> coeffMats = coefficient_matrices(basis, FN2, CN2);

  ML = Dec(coeffMats[0]);
  MR = Dec(coeffMats[1]);
  MCL = Dec(coeffMats[2]);
  MCR = Dec(coeffMats[3]);
  SIG = oscillation_indicator(basis);

  tmp = Mat(N, V);
  num = Mat(N, V);
  den = Vec(V);
}

void WenoSolver::coeffs_inner(Dec M, Matr dataBlock, double LAM) {

  tmp = M.solve(dataBlock);

  for (int i = 0; i < V; i++) {
    double p = tmp.col(i).transpose() * SIG * tmp.col(i) + EPS;
    double p2 = p * p;
    double p4 = p2 * p2;
    double p8 = p4 * p4;
    double o = LAM / p8;
    num.col(i) += o * tmp.col(i);
    den(i) += o;
  }
}

Vec WenoSolver::coeffs(Matr data) {
  // Calculate coefficients of basis polynomials and weights

  num.setZero();
  den.setZero();

  coeffs_inner(ML, data.block(0, 0, N, V), LAMS);
  coeffs_inner(MR, data.block(N - 1, 0, N, V), LAMS);

  if (N > 2) {
    coeffs_inner(MCL, data.block(FN2, 0, N, V), LAMC);

    if (N % 2 == 0) // Two central stencils (N>3)
      coeffs_inner(MCR, data.block(CN2, 0, N, V), LAMC);
  }

  tmp.array() = num.array().rowwise() / den.transpose().array();
  return VecMap(tmp.data(), N * V);
}

Mat WenoSolver::reconstruction(Matr ub) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of ub: (nx + 2(N-1)) * (ny + 2(N-1)) * V
  // Size of wh: nx * ny * N * N * V

  iVec shape = nX;

  Mat rec = ub;
  int ndim = nX.size();

  for (int d = 0; d < ndim; d++) {

    // shape is shape of rec at each iteration, = (m_1,m_2,...,m_ndim)
    // n1 = m_1 x m_2 x...x m_(d-1)
    // n2 = m_d - 2(N-1)
    // n3 = m_(d+1) x m_(d+2) x...x m_ndim

    int n1 = 1;
    for (int i = 0; i < d; i++)
      n1 *= shape(i);

    int n2 = shape(d) - 2 * (N - 1);

    int n3 = 1;
    for (int i = d + 1; i < ndim; i++)
      n3 *= shape(i);

    int n4 = std::pow(N, d);

    Mat rec2(n1 * n2 * n3 * n4 * N, V);

    int stride = n3 * n4 * V;

    for (int i1 = 0; i1 < n1; i1++)
      for (int i3 = 0; i3 < n3; i3++)
        for (int i4 = 0; i4 < n4; i4++) {

          int indu = ((i1 * shape(d) * n3 + i3) * n4 + i4) * V;
          int indw = ((i1 * n2 * n3 + i3) * n4 + i4) * N * V;

          MatMap ub0(rec.data() + indu, shape(d), V, OuterStride(stride));

          // Returns the WENO reconstruction of u using polynomials in x
          // Shape of ub0: (n2 + 2(N-1), V)
          // Shapw of wh: (n2 * N, V)

          for (int i = 0; i < n2; i++) {

            VecMap vh(rec2.data() + indw + i * N * stride, N * V);

            // eval to stop lazy evaluation, which causes nans
            vh = coeffs(ub0.block(i, 0, 2 * N - 1, V)).eval();
          }
        }

    rec = rec2;
    shape(d) -= 2 * (N - 1);
  }

  return rec;
}
