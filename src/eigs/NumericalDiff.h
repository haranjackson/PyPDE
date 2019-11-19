#ifndef NUMERICAL_DIFF_H
#define NUMERICAL_DIFF_H

#include "../types.h"
#include "eigen3/Core"
#include <cassert>

enum NumericalDiffMode { Forward, Central };

Mat df(Matr jac, void (*F)(double *, double *, int), Vecr q, int d,
       NumericalDiffMode mode) {

  const double eps = std::sqrt(Eigen::NumTraits<double>::epsilon());

  int n = q.size();
  Vec val1(n);
  Vec val2(n);

  switch (mode) {
  case Forward:
    F(val1.data(), q.data(), d);
    break;
  case Central:
    break;
  default:
    assert(false);
  };

  for (int i = 0; i < n; i++) {

    double qj = q(i);

    double h = std::max(eps * std::abs(qj), eps);

    switch (mode) {
    case Forward:

      q(i) += h;
      F(val2.data(), q.data(), d);

      q(i) = qj;
      jac.col(i) = (val2 - val1) / h;
      break;

    case Central:

      q(i) += h;
      F(val2.data(), q.data(), d);

      q(i) -= 2 * h;
      F(val1.data(), q.data(), d);

      q(i) = qj;
      jac.col(i) = (val2 - val1) / (2 * h);
      break;
    };
  }
  return jac;
}

#endif // NUMERICAL_DIFF_H
