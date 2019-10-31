extern "C" double test_func(void (*f)(double *, double *)) {

  double x[3] = {1., 2., 3.};
  double ret[3];
  f(x, ret);
  return ret[0];
}