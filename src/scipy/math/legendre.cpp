#include <vector>

#include "../../types.h"

Vec legval(Vec x, Vec c) {
  /*  Evaluate a Legendre series at points x.
      If `c` is of length `n + 1`, this function returns the value:

      p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

      `p(x)` has the same shape as `x`.
      Trailing zeros in the coefficients will be used in the evaluation, so
      they should be avoided if efficiency is a concern.

      Input
      ----------
      x : array

      c : array
      Array of coefficients ordered so that the coefficients for terms of
      degree n are contained in c[n].

      Output
      -------
      values : array

      Notes
      -----
      The evaluation uses Clenshaw recursion, aka synthetic division.
  */
  int nc = c.size();
  int n = x.size();
  Vec ret(n);
  ret.setZero(n);

  if (nc == 1) {
    ret.array() += c(0);
  } else if (nc == 2) {
    ret = c(0) + c(1) * x.array();
  } else {
    int nd = nc - 1;
    double c0 = c(nc - 3) - (c(nc - 1) * (nd - 1)) / nd;
    Vec c10 = c(nc - 2) + (c(nc - 1) * x.array() * (2 * nd - 1)) / nd;

    if (nc == 3) {
      ret = c0 + c10.array() * x.array();
    } else {
      nd -= 1;
      Vec c00 = c(nc - 4) - (c10.array() * (nd - 1)) / nd;
      Vec c11 = c0 + (c10.array() * x.array() * (2 * nd - 1)) / nd;

      for (int i = 5; i < nc + 1; i++) {
        Vec tmp = c00;
        nd -= 1;
        c00 = c(nc - i) - (c11.array() * (nd - 1)) / nd;
        c11 = tmp.array() + (c11.array() * x.array() * (2 * nd - 1)) / nd;
      }
      ret = c00.array() + c11.array() * x.array();
    }
  }
  return ret;
}

Mat legcompanion(Vec c) {
  /*  Return the scaled companion matrix of c.
      The basis polynomials are scaled so that the companion matrix is symmetric
      when `c` is an Legendre basis polynomial. This provides better eigenvalue
      estimates than the unscaled case and for basis polynomials the eigenvalues
      are guaranteed to be real if `numpy.linalg.eigvalsh` is used to obtain
      them.

      Input
      ----------
      c : array
      1-D array of Legendre series coefficients ordered from low to high degree.

      Output
      -------
      mat : array
      Scaled companion matrix of dimensions (deg, deg).
  */
  int n = c.size() - 1;
  Mat mat(n, n);
  mat.setZero(n, n);
  Vec scl(n);
  for (int i = 0; i < n; i++) {
    scl(i) = 1. / sqrt(2 * i + 1);
  }
  for (int i = 0; i < n - 1; i++) {
    double tmp = (i + 1) * scl(i) * scl(i + 1);
    mat(1 + i * (n + 1)) = tmp;
    mat(n + i * (n + 1)) = tmp;
  }
  return mat;
}

Vec legder(Vec c) { /*
                                      Differentiate a Legendre series.
                                      Returns the Legendre series coefficients
                       `c` differentiated once. he argument `c` is an array of
                       coefficients from low to high degree, e.g. [1,2,3]
                       represents the series ``1*L_0 + 2*L_1 + 3*L_2``.

                                      Input
                                      ----------
                                      c : array
                                          Array of Legendre series coefficients.

                                      Output
                                      -------
                                      der : array
                                          Legendre series of the derivative.

                                      Notes
                                      -----
                                      In general, the result of differentiating
                       a Legendre series does not resemble the same operation on
                       a power series. Thus the result of this function may be
                       "unintuitive," albeit correct.
                                      */

  int n = c.size() - 1;
  Vec der(n);
  der.setZero(n);
  for (int j = n; j > 2; j--) {
    der(j - 1) = (2 * j - 1) * c(j);
    c(j - 2) += c(j);
  }
  if (n > 1) {
    der(1) = 3 * c(2);
  }
  der(0) = c(1);
  return der;
}

std::vector<Vec> leggauss(int deg) {
  /*  Computes the nodes and weights for Gauss-Legendre quadrature.
      These nodes and weights will correctly integrate polynomials of
      degree < 2*deg over the interval [-1, 1] with the weight function
      w(x) = 1.

      Input
      ----------
      deg : int
      Number of sample points and weights (must be >= 1)

      Output
      -------
      x : array
      1D array containing the nodes

      w : array
      1D array containing the weights

      Notes
      -----
      The results have only been tested up to degree 100, higher degrees may
      be problematic. The weights are determined by using the fact that
      w_k = c / (L'_n(x_k) * L_{n-1}(x_k))
      where c is a constant independent of k and x_k is the kth root of L_n,
      and then scaling the results to get the right value when integrating 1.
  */

  // First approximation of roots. We use the fact that the companion
  // matrix is symmetric in this case in order to obtain better zeros.
  Vec c(deg + 1);
  c.setZero(deg + 1);
  c(deg) = 1;

  Mat m = legcompanion(c);
  Eigen::SelfAdjointEigenSolver<Mat> eigs(m);
  Vec x = eigs.eigenvalues();

  // Improve roots by one application of Newton.
  Vec dy = legval(x, c);
  Vec df = legval(x, legder(c));
  x -= (dy.array() / df.array()).matrix();

  // Compute the weights. Factor is scaled to avoid numerical overflow.
  Vec fm = legval(x, c.tail(deg));
  fm /= fm.lpNorm<Eigen::Infinity>();
  df /= df.lpNorm<Eigen::Infinity>();
  Vec w(deg);
  w = (1. / (fm.array() * df.array()));

  // Symmetrize
  w = (w + w.reverse()) / 2;
  x = (x - x.reverse()) / 2;

  // Scale w to get the right value
  w *= 2. / w.sum();

  std::vector<Vec> ret(2);
  ret[0] = x;
  ret[1] = w;
  return ret;
}
