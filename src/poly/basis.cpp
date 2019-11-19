#include <vector>

#include "../scipy/math/legendre.h"
#include "../scipy/math/polynomials.h"
#include "../types.h"

Vec scaled_nodes(int N) {
  std::vector<Vec> tmp = leggauss(N);
  Vec nodes = tmp[0];
  nodes.array() += 1;
  nodes /= 2;
  return nodes;
}

Vec scaled_weights(int N) {
  std::vector<Vec> tmp = leggauss(N);
  Vec wghts = tmp[1];
  wghts /= 2;
  return wghts;
}

poly lagrange(Vecr x, int i) {
  /*  Return a Lagrange interpolating polynomial at the ith Legendre node.
      Warning: This implementation is numerically unstable. Do not use more than
      about 20 points even if they are chosen optimally.

      Input
      ----------
      x : array
          Legendre nodes

      Output
      -------
      lagrange : numpy.poly1d instance
          The Lagrange interpolating polynomial.
  */
  poly p = poly((Vec(1) << 1).finished());
  for (int j = 0; j < x.size(); j++) {
    if (j == i)
      continue;
    p = p * poly((Vec(2) << -x(j), 1).finished()) / (x(i) - x(j));
  }
  return p;
}

std::vector<poly> basis_polys(int N) {
  // Returns basis polynomials
  Vec nodes = scaled_nodes(N);
  std::vector<poly> psi(N);
  for (int i = 0; i < N; i++)
    psi[i] = lagrange(nodes, i);
  return psi;
}

Mat end_values(const std::vector<poly> &basis) {
  // ret[i,0], ret[i,1] are the the values of ith basis polynomial at 0,1

  int N = basis.size();
  Mat ret(2, N);

  for (int j = 0; j < N; j++) {
    ret(0, j) = basis[j].eval(0.);
    ret(1, j) = basis[j].eval(1.);
  }
  return ret;
}

Mat derivative_values(const std::vector<poly> &basis, Vecr NODES) {
  // ret[i,j] is the derivative of the jth basis function at the ith node
  int N = basis.size();
  Mat ret(N, N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      ret(i, j) = basis[j].diff(1).eval(NODES(i));
  return ret;
}
