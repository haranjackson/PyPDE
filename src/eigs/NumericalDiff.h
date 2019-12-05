#ifndef NUMERICAL_DIFF_H
#define NUMERICAL_DIFF_H

#include "../types.h"

void df(Matr jac, void (*F)(double *, double *, double *, int), Vecr q, Matr dq,
        int d, bool secondOrder, bool forwardMode = true) {

  const double eps = std::sqrt(Eigen::NumTraits<double>::epsilon());

  int V = q.size();
  Vec val1(V);
  Vec val2(V);

  if (forwardMode)
    F(val1.data(), q.data(), dq.data(), d);

  for (int i = 0; i < V; i++) {

    double *qi = secondOrder ? dq.data() + d * V + i : q.data() + i;
    double qi0 = *qi;

    double h = std::max(eps * std::abs(qi0), eps);

    if (forwardMode) {

      *qi += h;
      F(val2.data(), q.data(), dq.data(), d);
      *qi = qi0;

      jac.col(i) = (val2 - val1) / h;

    } else {

      *qi += h;
      F(val2.data(), q.data(), dq.data(), d);
      *qi -= 2 * h;
      F(val1.data(), q.data(), dq.data(), d);
      *qi = qi0;

      jac.col(i) = (val2 - val1) / (2 * h);
    }
  }
}

#endif // NUMERICAL_DIFF_H
