#ifndef NUMERICAL_DIFF_H
#define NUMERICAL_DIFF_H

#include "eigen3/Core"
#include "types.h"
#include <cassert>

enum NumericalDiffMode { Forward, Central };

Mat df(void (*F)(double *, double *, int), Vecr q, int d,
       NumericalDiffMode mode) {

  using std::abs;
  using std::sqrt;

  double h;
  const double eps =
      sqrt(((std::max)(0., Eigen::NumTraits<double>::epsilon())));

  int n = q.size();
  Mat jac(n, n);
  Vec val1(n);
  Vec val2(n);
  Vec q_ = q;

  switch (mode) {
  case Forward:
    F(val1.data(), q_.data(), d);
    break;
  case Central:
    break;
  default:
    assert(false);
  };

  for (int j = 0; j < n; ++j) {
    h = eps * abs(q_[j]);
    if (h == 0.) {
      h = eps;
    }
    switch (mode) {
    case Forward:

      q_[j] += h;
      F(val2.data(), q_.data(), d);

      q_[j] = q[j];
      jac.col(j) = (val2 - val1) / h;
      break;

    case Central:

      q_[j] += h;
      F(val2.data(), q_.data(), d);

      q_[j] -= 2 * h;
      F(val1.data(), q_.data(), d);

      q_[j] = q[j];
      jac.col(j) = (val2 - val1) / (2 * h);
      break;

    default:
      assert(false);
    };
  }
  return jac;
}

#endif // NUMERICAL_DIFF_H
