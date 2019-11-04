#include "indexing.h"
#include "types.h"

const int TRANSMISSIVE = 0;
const int PERIODIC = 1;

int renormalised_index(int ind, int N, int type, int bound) {

  int ret = ind - N;

  if (ret < 0) {
    if (type == TRANSMISSIVE)
      return 0;
    if (type == PERIODIC)
      return bound + ret;
  }
  if (ret >= bound) {
    if (type == TRANSMISSIVE)
      return bound - 1;
    if (type == PERIODIC)
      return ret - bound;
  }
  return ret;
}

Vec boundaries(Vecr u, iVecr nX, iVecr boundaryTypes, int N) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions

  int ndim = nX.size();
  int V = u.size() / nX.prod();

  iVec nX_2N = nX;

  nX_2N.array() += 2 * N;
  int ncell = nX_2N.prod();

  Vec ub(ncell);

  iVec inds = iVec::Zero(ndim);
  iVec tmpInds(ndim);

  for (int idx = 0; idx < ncell; idx++) {

    update_inds(inds, nX_2N);

    for (int d = 0; d < ndim; d++) {
      tmpInds(d) = renormalised_index(inds(d), N, boundaryTypes(d), nX(d));
    }

    ub.segment(idx * V, V) = u.segment(index(tmpInds, nX, 0), V);
  }

  return ub;
}
