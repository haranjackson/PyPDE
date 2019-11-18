#ifndef ADER_H
#define ADER_H

extern "C" void ader_solver(void (*F)(double *, double *, int),
                            void (*B)(double *, double *, int),
                            void (*S)(double *, double *), bool useF, bool useB,
                            bool useS, double *_u, double tf, int *_nX,
                            int ndim, double *_dX, double CFL,
                            int *_boundaryTypes, bool STIFF, int FLUX, int N,
                            int V, int ndt, double *_ret);

extern "C" void weno_solver(double *ret, double *_u, int *_nX, int ndim, int N,
                            int V);

#endif