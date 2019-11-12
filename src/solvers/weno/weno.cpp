#include "../../etc/types.h"
#include "../poly/basis.h"
#include "weno_matrices.h"
#include <iostream>

const double LAMS = 1.;   // WENO side stencil weighting
const double LAMC = 1e5;  // WENO central stencil weighting
const double EPS = 1e-14; // WENO epsilon parameter

void weight(Vecr ret, Matr w, double LAM, Matr SIG, int V) {
  // Produces the WENO weight for this stencil
  // NOTE: The denominator is raised to the 8th power.
  //       The square method is used because pow is slow.
  for (int i = 0; i < V; i++) {
    double tmp = w.col(i).transpose() * SIG * w.col(i) + EPS;
    double tmp2 = tmp * tmp;
    double tmp4 = tmp2 * tmp2;
    double den = tmp4 * tmp4;
    ret(i) = LAM / den;
  }
}

void coeffs_inner(Vecr o, Matr w, int V, Dec M, Matr dataBlock, Matr SIG,
                  Matr num, Vecr den, double LAM) {

  w = M.solve(dataBlock);
  weight(o, w, LAM, SIG, V);
  num.array() += w.array().rowwise() * o.transpose().array();
  den += o;
}

void coeffs(Matr ret, Matr data, int N, int V, int FN2, int CN2, Dec ML, Dec MR,
            Dec MCL, Dec MCR, Matr SIG) {
  // Calculate coefficients of basis polynomials and weights

  Vec o(V);
  Mat w(N, V);

  Mat num = Mat::Zero(N, V);
  Vec den = Vec::Zero(V);

  coeffs_inner(o, w, V, ML, data.block(0, 0, N, V), SIG, num, den, LAMS);
  coeffs_inner(o, w, V, MR, data.block(N - 1, 0, N, V), SIG, num, den, LAMS);

  if (N > 2) {
    coeffs_inner(o, w, V, MCL, data.block(FN2, 0, N, V), SIG, num, den, LAMC);

    if (N % 2 == 0) // Two central stencils (N>3)
      coeffs_inner(o, w, V, MCR, data.block(CN2, 0, N, V), SIG, num, den, LAMC);
  }

  ret.array() = num.array().rowwise() / den.transpose().array();
}

Mat weno1(Matr ub, int nx, int N, int V, int FN2, int CN2, Dec ML, Dec MR,
          Dec MCL, Dec MCR, Matr SIG) {
  // Returns the WENO reconstruction of u using polynomials in x
  // Shape of ub: (nx + 2(N-1), V)
  // Shapw of wh: (nx * N, V)

  Mat wh(nx * N, V);

  for (int i = 0; i < nx; i++)

    coeffs(wh.block(i * N, 0, N, V), ub.block(i, 0, 2 * N - 1, V), N, V, FN2,
           CN2, ML, MR, MCL, MCR, SIG);

  return wh;
}

Mat weno_inner(Matr ub, iVecr nX, int N, int V, int FN2, int CN2, Dec ML,
               Dec MR, Dec MCL, Dec MCR, Matr SIG) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of ub: (nx + 2(N-1)) * (ny + 2(N-1)) * V
  // Size of wh: nx * ny * N * N * V

  iVec shape = nX;
  shape.array() += 2 * (N - 1);

  Mat rec = ub;
  int ndim = nX.size();

  for (int d = 0; d < ndim; d++) {

    int n1 = 1;
    for (int i = 0; i < d; i++)
      n1 *= shape(i);

    int n2 = shape(d) - 2 * (N - 1);

    int n3 = 1;
    for (int i = d + 1; i < ndim; i++)
      n3 *= shape(i);

    int n4 = pow(N, d);

    Mat tmp = Mat::Zero(n1 * n2 * n3 * n4 * N, V);

    int stride = n3 * n4 * V;

    for (int i1 = 0; i1 < n1; i1++)
      for (int i3 = 0; i3 < n3; i3++)
        for (int i4 = 0; i4 < n4; i4++) {

          int indu = (i1 * shape(d) * n3 + i3) * n4 + i4;
          int indw = (i1 * n2 * n3 + i3) * n4 + i4;

          MatMap ub0(rec.data() + indu, shape(d), V, OuterStride(stride));

          Mat wh0 =
              MatMap(tmp.data() + indw, n2, N * V, OuterStride(N * stride));

          Mat wh = weno1(ub0, n2, N, V, FN2, CN2, ML, MR, MCL, MCR, SIG);

          wh0 = MatMap(wh.data(), n2, N * V, OuterStride(N * V));
        }

    rec = tmp;
    shape(d) -= 2 * (N - 1);
  }

  return rec;
}

Mat weno_reconstruction(Matr ub, iVecr nX, int N, int V) {

  int FN2 = (int)floor((N - 1) / 2.);
  int CN2 = (int)ceil((N - 1) / 2.);

  std::vector<poly> basis = basis_polys(N);

  std::vector<Mat> coeffMats = coefficient_matrices(basis, FN2, CN2);
  Dec ML(coeffMats[0]);
  Dec MR(coeffMats[1]);
  Dec MCL(coeffMats[2]);
  Dec MCR(coeffMats[3]);
  Mat SIG = oscillation_indicator(basis);

  return weno_inner(ub, nX, N, V, FN2, CN2, ML, MR, MCL, MCR, SIG);
}
