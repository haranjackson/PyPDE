#include <iostream>

#include "../../types.h"
#include "lgmres.h"
#include "newton_krylov.h"

double maxnorm(Vecr x) { return x.lpNorm<Eigen::Infinity>(); }

void KrylovJacobian::update_diff_step() {
  double mx = maxnorm(x0);
  double mf = maxnorm(f0);
  omega = rdiff * std::max(1., mx) / std::max(1., mf);
}

KrylovJacobian::KrylovJacobian(Vecr x, Vecr f, VecFunc F) {
  func = F;
  maxiter = 1;
  inner_m = 30;
  outer_k = 10;

  x0 = x;
  f0 = f;
  rdiff = std::pow(mEPS, 0.5);
  update_diff_step();
  outer_v = std::vector<Vec>(0);
}

Vec KrylovJacobian::matvec(Vecr v) {
  double nv = v.norm();
  if (nv == 0.)
    return 0. * v;
  double sc = omega / nv;
  Vec tmp = x0 + sc * v;
  return (func(tmp) - f0) / sc;
}

Vec KrylovJacobian::psolve(Vecr v) { return v; }

Vec KrylovJacobian::solve(Vecr rhs, double tol) {
  Vec x0 = 0. * rhs;
  using std::placeholders::_1;
  std::function<Vec(Vecr)> mvec = std::bind(&KrylovJacobian::matvec, *this, _1);
  std::function<Vec(Vecr)> psol = std::bind(&KrylovJacobian::psolve, *this, _1);

  return lgmres(mvec, psol, rhs, x0, outer_v, tol, maxiter, inner_m, outer_k);
}

void KrylovJacobian::update(Vecr x, Vecr f) {
  x0 = x;
  f0 = f;
  update_diff_step();
}

TerminationCondition::TerminationCondition(double ftol, double frtol,
                                           double xtol, double xrtol) {
  x_tol = xtol;
  x_rtol = xrtol;
  f_tol = ftol;
  f_rtol = frtol;
  f0_norm = 0.;
}

int TerminationCondition::check(Vecr f, Vecr x, Vecr dx) {
  double f_norm = maxnorm(f);
  double x_norm = maxnorm(x);
  double dx_norm = maxnorm(dx);

  if (f0_norm == 0.)
    f0_norm = f_norm;

  if (f_norm == 0.)
    return 1;

  return int((f_norm <= f_tol && f_norm / f_rtol <= f0_norm) &&
             (dx_norm <= x_tol && dx_norm / x_rtol <= x_norm));
}

double phi(double s, double *tmp_s, double *tmp_phi, Vecr tmp_Fx, VecFunc func,
           Vecr x, Vecr dx) {
  if (s == *tmp_s)
    return *tmp_phi;
  Vec xt = x + s * dx;
  Vec v = func(xt);
  double p = v.squaredNorm();
  *tmp_s = s;
  *tmp_phi = p;
  tmp_Fx = v;
  return p;
}

double scalar_search_armijo(double phi0, double *tmp_s, double *tmp_phi,
                            Vecr tmp_Fx, VecFunc func, Vecr x, Vecr dx) {
  double c1 = 1e-4;
  double amin = 1e-2;

  double phi_a0 = phi(1, tmp_s, tmp_phi, tmp_Fx, func, x, dx);
  if (phi_a0 <= phi0 - c1 * phi0)
    return 1.;

  double alpha1 = phi0 / (2. * phi_a0);
  double phi_a1 = phi(alpha1, tmp_s, tmp_phi, tmp_Fx, func, x, dx);

  if (phi_a1 <= phi0 - c1 * alpha1 * phi0)
    return alpha1;

  while (alpha1 > amin) {
    double factor = alpha1 * alpha1 * (alpha1 - 1);
    double a = phi_a1 - phi0 + phi0 * alpha1 - alpha1 * alpha1 * phi_a0;
    a /= factor;
    double b =
        -(phi_a1 - phi0 + phi0 * alpha1) + alpha1 * alpha1 * alpha1 * phi_a0;
    b /= factor;

    double alpha2 = (-b + sqrt(std::abs(b * b + 3 * a * phi0))) / (3. * a);
    double phi_a2 = phi(alpha2, tmp_s, tmp_phi, tmp_Fx, func, x, dx);

    if (phi_a2 <= phi0 - c1 * alpha2 * phi0)
      return alpha2;

    if ((alpha1 - alpha2) > alpha1 / 2.0 || (1 - alpha2 / alpha1) < 0.96)
      alpha2 = alpha1 / 2.0;

    alpha1 = alpha2;
    phi_a0 = phi_a1;
    phi_a1 = phi_a2;
  }
  return 1.;
}

void _nonlin_line_search(VecFunc func, Vecr x, Vecr Fx, Vecr dx) {
  double tmp_s = 0.;
  double tmp_phi = Fx.squaredNorm();
  Vec tmp_Fx = Fx;

  double s =
      scalar_search_armijo(tmp_phi, &tmp_s, &tmp_phi, tmp_Fx, func, x, dx);
  x += s * dx;
  if (s == tmp_s)
    Fx = tmp_Fx;
  else
    Fx = func(x);
}

Vec nonlin_solve(VecFunc F, Vecr x, double f_tol, double f_rtol, double x_tol,
                 double x_rtol) {
  TerminationCondition condition(f_tol, f_rtol, x_tol, x_rtol);

  double gamma = 0.9;
  double eta_max = 0.9999;
  double eta_treshold = 0.1;
  double eta = 1e-3;

  Vec dx = INF * Vec::Ones(x.size());
  Vec Fx = F(x);
  double Fx_norm = maxnorm(Fx);

  KrylovJacobian jacobian(x, Fx, F);

  int maxiter = 3 * (x.size() + 1);

  for (int n = 0; n < maxiter; n++) {

    if (condition.check(Fx, x, dx))
      break;

    double tol = std::min(eta, eta * Fx_norm);
    Vec dx = -jacobian.solve(Fx, tol);

    _nonlin_line_search(F, x, Fx, dx);
    double Fx_norm_new = Fx.norm();

    jacobian.update(x, Fx);

    double eta_A = gamma * Fx_norm_new * Fx_norm_new / (Fx_norm * Fx_norm);
    if (gamma * eta * eta < eta_treshold)
      eta = std::min(eta_max, eta_A);
    else
      eta = std::min(eta_max, std::max(eta_A, gamma * eta * eta));

    Fx_norm = Fx_norm_new;
  }
  return x;
}
