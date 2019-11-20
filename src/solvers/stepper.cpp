#include "stepper.h"
#include "../eigs/system.h"
#include "../grid/indexing.h"
#include "../poly/basis.h"
#include "../poly/evaluations.h"
#include "../types.h"

TimeStepper::TimeStepper(void (*_F)(double *, double *, double *, int),
                         void (*_B)(double *, double *, int), Vecr _dX, int _N,
                         int _V, double _CFL, double _tf, bool _secondOrder)
    : F(_F), B(_B), dX(_dX), N(_N), V(_V), CFL(_CFL), tf(_tf),
      secondOrder(_secondOrder) {

  ndim = dX.size();
  Nd = std::pow(N, ndim);

  std::vector<poly> basis = basis_polys(N);
  Vec NODES = scaled_nodes(N);

  WGHTS = scaled_weights(N);
  DERVALS = derivative_values(basis, NODES);

  WGHT_PRODS = Vec::Ones(Nd);
  iVec inds = iVec::Zero(ndim);

  for (int idx = 0; idx < Nd; idx++) {

    for (int d = 0; d < ndim; d++)
      WGHT_PRODS(idx) *= WGHTS(inds(d));

    update_inds(inds, N);
  }
};

double TimeStepper::step(Matr wh, double t, int count) {

  Vec q(V);
  Mat dwh0(Nd, V);
  Mat dq(ndim, V);

  double MAX = 0.;
  for (int ind = 0; ind < wh.rows(); ind += Nd) {

    MatMap wh0(wh.data() + ind * V, Nd, V, OuterStride(V));

    q = WGHT_PRODS.transpose() * wh0;

    for (int d = 0; d < ndim; d++) {
      derivs(dwh0, wh0, d, DERVALS, ndim, dX);
      dq.row(d) = WGHT_PRODS.transpose() * dwh0;
    }

    double tmp = 0.;
    for (int d = 0; d < ndim; d++) {

      double lam = max_abs_eigs(F, B, q, dq, d);

      if (secondOrder)
        lam += max_abs_eigs_second_order(F, q, dq, d, N, dX);

      tmp += lam / dX(d);
    }

    MAX = std::max(MAX, tmp);
  }

  double dt = CFL / MAX;

  if (count <= 5)
    dt *= 0.2;

  if (t + dt > tf)
    return tf - t;

  return dt;
}