#include "../types.h"
#include "indexing.h"

const int TRANSMISSIVE = 0;
const int PERIODIC = 1;

int renormalised_index(int ind, int N, int type, int bound) {

  int ret = ind - N;

  if (ret < 0) {
    if (type == TRANSMISSIVE)
      return 0;
    else if (type == PERIODIC)
      return bound + ret;
  } else if (ret >= bound) {
    if (type == TRANSMISSIVE)
      return bound - 1;
    else if (type == PERIODIC)
      return ret - bound;
  }
  return ret;
}

Mat boundaries(Matr u, iVecr nX, iVecr boundaryTypes, int N) {
  // If periodic is true, applies periodic boundary conditions,
  // else applies transmissive boundary conditions

  int ndim = nX.size();
  int V = u.size() / nX.prod();

  iVec nX2N = nX;
  nX2N.array() += 2 * N;

  int ncell = nX2N.prod();

  Mat ub(ncell, V);

  iVec inds = iVec::Zero(ndim);
  iVec tmpInds(ndim);

  for (int idx = 0; idx < ncell; idx++) {

    for (int d = 0; d < ndim; d++)
      tmpInds(d) = renormalised_index(inds(d), N, boundaryTypes(d), nX(d));

    ub.row(idx) = u.row(index(tmpInds, nX, 0));

    update_inds(inds, nX2N);
  }

  return ub;
}
