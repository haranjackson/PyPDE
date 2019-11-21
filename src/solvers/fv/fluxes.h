#ifndef FLUXES_H
#define FLUXES_H

#include "../../types.h"

const int RUSANOV = 0;
const int ROE = 1;
const int OSHER = 2;

class FluxGenerator {
private:
  void (*F)(double *, double *, double *, int);
  void (*B)(double *, double *, int);

  Vec NODES;
  Vec WGHTS;
  Vec dX;

  int N;
  int V;

  int FLUX;

  Vec fL;
  Vec fR;
  Vec q;
  Mat dq;
  Mat M;

  Eigen::EigenSolver<Mat> ES;

  Vec D_OSH(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d);

  Vec D_ROE(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d);

  Vec D_RUS(Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d);

public:
  FluxGenerator(void (*F)(double *, double *, double *, int),
                void (*B)(double *, double *, int), Vecr NODES, Vecr WGHTS,
                Vecr dX, int V, int FLUX, int ndim);

  void flux(Vecr ret, Vecr qL, Vecr qR, Matr dqL, Matr dqR, int d,
            bool secondOrder);

  void Bint(Vecr ret, Vecr qL, Vecr qR, int d);
};

#endif // FLUXES_H
