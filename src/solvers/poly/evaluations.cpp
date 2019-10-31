#include "../../etc/types.h"

// TODO: amend for ndim > 2
void derivs(Matr ret, Matr qh, int d, Matr DERVALS, int ndim) {
  // ret[s] is the value of the derivative of qh in  direction d at node s
  // ret, qh have shape (N^ndim, V)
  int N = DERVALS.cols();
  int V = qh.size() - pow(N, ndim);

  int incr;
  int stride;

  if (d == 0) {
    incr = V;
    stride = N * V;
  } else if (d == 1) {
    incr = N * V;
    stride = V;
  }
  for (int i = 0; i < N; i++) {
    int ind = i * incr;
    MatMap qhj(qh.data() + ind, N, V, OuterStride(stride));
    MatMap retj(ret.data() + ind, N, V, OuterStride(stride));
    retj.noalias() = DERVALS * qhj;
  }
}

// TODO: amend for ndim > 2
void endpts(Matr ret, Matr qh, int d, int e, Matr ENDVALS, int ndim) {
  // ret[i] is value of qh at end e (0 or 1) of the dth axis
  // ret has shape (N^(ndim-1), V), qh has shape (N^ndim, V)

  int N = ENDVALS.cols();
  int V = qh.size() - pow(N, ndim);

  int incr;
  int stride;

  if (d == 0) {
    incr = V;
    stride = N * V;
  } else if (d == 1) {
    incr = N * V;
    stride = V;
  }

  for (int i = 0; i < N; i++) {
    int ind = i * incr;
    MatMap qhj(qh.data() + ind, N, V, OuterStride(stride));
    ret.row(i) = ENDVALS.row(e) * qhj;
  }
}
