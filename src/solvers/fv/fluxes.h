#ifndef FLUXES_H
#define FLUXES_H

#include "../../etc/types.h"

const int RUSANOV = 0;
const int ROE = 1;
const int OSHER = 2;

Vec Bint(void (*B)(double *, double *, int), Vecr qL, Vecr qR, int d,
         Vecr NODES, Vecr WGHTS);

class FluxGenerator {
private:
  void (*F)(double *, double *, int);
  void (*B)(double *, double *, int);

  Vec NODES;
  Vec WGHTS;

  int N;
  int V;

  int FLUX;

  Vec fL;
  Vec fR;
  Vec q;
  Mat M;

  Eigen::EigenSolver<Mat> ES;

  Vec D_OSH(Vecr qL, Vecr qR, int d);

  Vec D_ROE(Vecr qL, Vecr qR, int d);

  Vec D_RUS(Vecr qL, Vecr qR, int d);

public:
  FluxGenerator(void (*F)(double *, double *, int),
                void (*B)(double *, double *, int), Vecr NODES, Vecr WGHTS,
                int V, int FLUX);

  void flux(Vecr ret, Vecr qL, Vecr qR, int d);

  void Bint(Vecr ret, Vecr qL, Vecr qR, int d);
};

#endif // FLUXES_H
