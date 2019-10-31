#include "types.h"

const int TRANSMISSIVE = 0;
const int PERIODIC = 1;

void boundaries1(Matr u, Matr ub, int nx, iVecr boundaryTypes, int d, int N,
                 int V) {

  switch (boundaryTypes(0)) {

  case TRANSMISSIVE:
    for (int i = 0; i < N; i++)
      ub.row(i) = u.row(0);
    break;

  case PERIODIC:
    for (int i = 0; i < N; i++)
      ub.row(i) = u.row(nx - N + i);
    break;
  }

  switch (boundaryTypes(1)) {

  case TRANSMISSIVE:
    for (int i = 0; i < N; i++)
      ub.row(i + nx + N) = u.row(nx - 1);
    break;

  case PERIODIC:
    for (int i = 0; i < N; i++)
      ub.row(i + nx + N) = u.row(i);
    break;
  }
}

void boundaries2(Vecr u, Vecr ub, int nx, int ny, iVecr boundaryTypes, int N,
                 int V) {

  // (i * ny + j) * V;
  // (i * (ny + 2 * N) + j) * V;

  for (int j = 0; j < ny; j++) {
    MatMap u0(u.data() + (j * V), nx, V, OuterStride(ny * V));
    MatMap ub0(ub.data() + ((j + N) * V), nx + 2 * N, V,
               OuterStride((ny + 2 * N) * V));
    boundaries1(u0, ub0, nx, boundaryTypes.head<2>(), 0);
  }

  for (int i = 0; i < nx + 2 * N; i++) {
    MatMap u0(ub.data() + ((i * (ny + 2 * N) + N) * V), ny, V, OuterStride(V));
    MatMap ub0(ub.data() + ((i * (ny + 2 * N)) * V), ny + 2 * N, V,
               OuterStride(V));
    boundaries1(u0, ub0, ny, boundaryTypes.tail<2>(), 1);
  }
}

void boundaries(Vecr u, Vecr ub, iVecr nX, iVecr boundaryTypes, int N, int V) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions
  long ndim = nX.size();
  int nx = nX(0);

  switch (ndim) {

  case 1: {
    ub.segment(N * V, nx * V) = u;
    MatMap u0(u.data(), nx, V, OuterStride(V));
    MatMap ub0(ub.data(), nx + 2 * N, V, OuterStride(V));
    boundaries1(u0, ub0, nx, boundaryTypes, 0, N, V);
  } break;

  case 2:
    int ny = nX(1);
    for (int i = 0; i < nx; i++)
      ub.segment(((i + N) * (ny + 2 * N) + N) * V, ny * V) =
          u.segment(i * ny * V, ny * V);
    boundaries2(u, ub, nx, ny, boundaryTypes, N, V);
    break;
  }
}

int extended_dimensions(iVecr nX, int ext) {
  return (nX.array() + 2 * ext).prod();
}
