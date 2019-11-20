#ifndef STEPPER_H
#define STEPPER_H

#include "../types.h"

class TimeStepper {
private:
  void (*F)(double *, double *, double *, int);
  void (*B)(double *, double *, int);

  Vec dX;
  int ndim;

  int N;
  int V;
  int Nd;

  double CFL;
  double tf;
  bool secondOrder;

  Vec WGHTS;
  Mat DERVALS;
  Vec WGHT_PRODS;

public:
  TimeStepper(void (*_F)(double *, double *, double *, int),
              void (*_B)(double *, double *, int), Vecr _dX, int _N, int _V,
              double _CFL, double _tf, bool _secondOrder);

  double step(Matr wh, double t, int count);
};

#endif