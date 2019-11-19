#include <cmath>

#include "../../types.h"
#include "polynomials.h"

Vec integrate(Vec p) {
  // Return the integral of polynomial p, with 0 constant term
  int n = p.size();
  Vec ret(n + 1);
  ret(0) = 0;
  for (int i = 1; i < n + 1; i++)
    ret(i) = p(i - 1) / i;
  return ret;
}

Vec differentiate(Vec p) {
  // Return the derivative of polynomial p
  int n = p.size();
  Vec ret(n - 1);
  for (int i = 0; i < n - 1; i++)
    ret(i) = p(i + 1) * (i + 1);
  return ret;
}

double evaluate(Vec p, double x) {
  // Evaluate polynomial p at point x
  int n = p.size();
  double ret = 0;
  for (int i = 0; i < n; i++)
    ret += p(i) * std::pow(x, i);
  return ret;
}

Vec multiply(Vec p1, Vec p2) {
  // Multiply polynomials p1 and p2
  int n1 = p1.size();
  int n2 = p2.size();
  Vec ret(n1 + n2 - 1);
  ret.setZero(n1 + n2 - 1);
  for (int i = 0; i < n1; i++)
    for (int j = 0; j < n2; j++)
      ret(i + j) += p1(i) * p2(j);
  return ret;
}

poly::poly() {
  Vec c;
  coef = c;
}

poly::poly(Vec c) { coef = c; }

poly poly::intt() const { return poly(integrate(coef)); }

poly poly::diff(int n) const {
  Vec ret = coef;
  for (int i = 0; i < n; i++)
    ret = differentiate(ret);
  return ret;
}

double poly::eval(double x) const { return evaluate(coef, x); }

poly poly::operator/(double c) { return poly(coef / c); }

poly poly::operator*(const poly &p) { return poly(multiply(coef, p.coef)); }
