#include "../../etc/types.h"
#include "../poly/basis.h"
#include "weno_matrices.h"

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

void coeffs(Matr ret, Matr data, int nx, int ind, int N, int V, int FN2,
            int CN2, Dec ML, Dec MR, Dec MCL, Dec MCR, Matr SIG) {
  // Calculate coefficients of basis polynomials and weights

  Vec oL(V);
  Vec oR(V);
  Vec oCL(V);
  Vec oCR(V);

  Mat wL(N, V);
  Mat wR(N, V);
  Mat wCL(N, V);
  Mat wCR(N, V);

  wL = ML.solve(data.block(0, 0, N, V));
  wR = MR.solve(data.block(N - 1, 0, N, V));

  weight(oL, wL, LAMS, SIG, V);
  weight(oR, wR, LAMS, SIG, V);

  if (N > 2) {
    wCL = MCL.solve(data.block(FN2, 0, N, V));
    weight(oCL, wCL, LAMC, SIG, V);

    if (N % 2 == 0) // Two central stencils (N>3)
    {
      wCR = MCR.solve(data.block(CN2, 0, N, V));
      weight(oCR, wCR, LAMC, SIG, V);
    }
  }

  Mat num = Mat::Zero(N, V);
  Vec den = Vec::Zero(V);

  num.array() += wL.array().rowwise() * oL.transpose().array();
  den += oL;

  num.array() += wR.array().rowwise() * oR.transpose().array();
  den += oR;

  if (N > 2) {

    num.array() += wCL.array().rowwise() * oCL.transpose().array();
    den += oCL;

    if (N % 2 == 0) {
      num.array() += wCR.array().rowwise() * oCR.transpose().array();
      den += oCR;
    }
  }
  ret.array() = num.array().rowwise() / den.transpose().array();
}

void weno1(Matr wh, Matr ub, int nx, int N, int V, int FN2, int CN2, Dec ML,
           Dec MR, Dec MCL, Dec MCR, Matr SIG) {
  // Returns the WENO reconstruction of u using polynomials in x
  // Shape of ub: (nx + 2(N-1), V)
  // Shapw of wh: (nx * N, V)

  for (int i = 0; i < nx; i++)

    coeffs(wh.block(i * N, 0, N, V), ub.block(i, 0, 2 * N - 1, V), nx, i, N, V,
           FN2, CN2, ML, MR, MCL, MCR, SIG);
}

void weno2(Matr wh, Matr ub, int nx, int ny, int N, int V, int FN2, int CN2,
           Dec ML, Dec MR, Dec MCL, Dec MCR, Matr SIG) {
  // Returns the WENO reconstruction of u using polynomials in y
  // Size of ub: (nx + 2(N-1)) * (ny + 2(N-1)) * V
  // Size of wh: nx * ny * N * N * V

  Vec ux(nx * (ny + 2 * (N - 1)) * N * V);

  Mat tmp0(nx * N, V);
  MatMap tmp0Map(tmp0.data(), nx, N * V, OuterStride(N * V));

  for (int j = 0; j < ny + 2 * (N - 1); j++) {
    MatMap ubMap(ub.data() + j * V, nx + 2 * (N - 1), V,
                 OuterStride((ny + 2 * (N - 1)) * V));
    MatMap uxMap(ux.data() + j * N * V, nx, N * V,
                 OuterStride((ny + 2 * (N - 1)) * N * V));

    weno1(tmp0, ubMap, nx, N, V, FN2, CN2, ML, MR, MCL, MCR, SIG);
    uxMap = tmp0Map;
  }

  Mat tmp1(ny * N, V);
  MatMap tmp1Map(tmp1.data(), ny, N * V, OuterStride(N * V));

  for (int i = 0; i < nx; i++)
    for (int ii = 0; ii < N; ii++) {
      MatMap uxMap(ux.data() + (i * (ny + 2 * (N - 1)) * N + ii) * V,
                   (ny + 2 * (N - 1)), V, OuterStride(N * V));
      MatMap whMap(wh.data() + (i * ny * N + ii) * N * V, ny, N * V,
                   OuterStride(N * N * V));

      weno1(tmp1, uxMap, ny, N, V, FN2, CN2, ML, MR, MCL, MCR, SIG);
      whMap = tmp1Map;
    }
  // TODO: make this n-dimensional by turning these two sets of for loops into
  // n sets of for loops, contained within a large loop over the dimensions
}

void weno_launcher(Vecr wh, Vecr ub, iVecr nX, int N, int V) {
  // NOTE: boundary conditions extend u by two cells in each dimension

  int ndim = nX.size();
  int FN2 = (int)floor((N - 1) / 2.);
  int CN2 = (int)ceil((N - 1) / 2.);

  int nwh = wh.size() / V;
  int nub = ub.size() / V;

  std::vector<poly> basis = basis_polys(N);

  std::vector<Mat> coeffMats = coefficient_matrices(basis, FN2, CN2);
  Dec ML(coeffMats[0]);
  Dec MR(coeffMats[1]);
  Dec MCL(coeffMats[2]);
  Dec MCR(coeffMats[3]);
  Mat SIG = oscillation_indicator(basis);

  Vec ub_ = ub;

  MatMap whMap(wh.data(), nwh, V, OuterStride(V));
  MatMap ubMap(ub_.data(), nub, V, OuterStride(V));

  switch (ndim) {

  case 1:
    weno1(whMap, ubMap, nX(0) + 2, N, V, FN2, CN2, ML, MR, MCL, MCR, SIG);
    break;

  case 2:
    weno2(whMap, ubMap, nX(0) + 2, nX(1) + 2, N, V, FN2, CN2, ML, MR, MCL, MCR,
          SIG);
    break;
  }
}
