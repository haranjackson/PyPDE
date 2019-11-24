#include "../grid/indexing.h"
#include "../types.h"

int zero_index(iVecr inds, int N, int d) {
  // let inds = (i1, i2, ..., in) where 0 <= i < N for all i.
  // returns the index of cell (i1, i2, ..., i(d-1), 0, i(d+1), ..., in)
  int ndim = inds.size() + 1;
  int ret = 0;

  for (int i = 0; i < d; i++) {
    ret *= N;
    ret += inds(i);
  }
  ret *= std::pow(N, ndim - d);
  for (int i = d + 1; i < ndim; i++) {
    ret *= N;
    ret += inds(i - 1);
  }

  return ret;
}

void derivs(Matr ret, Matr qh, int d, Matr DERVALS, int ndim, Vecr dX) {
  // ret[s] is the value of the derivative of qh in  direction d at node s
  // ret, qh have shape (N^ndim, V)
  // NOTE: ret, qh must be contiguous

  int N = DERVALS.cols();
  int V = qh.cols();
  int Nd_ = std::pow(N, ndim - 1);

  iVec inds = iVec::Zero(ndim - 1);
  int stride = std::pow(N, ndim - d - 1) * V;

  for (int ind = 0; ind < Nd_; ind++) {

    int ind_ = zero_index(inds, N, d);

    MatMap qh_(qh.data() + ind_ * V, N, V, OuterStride(stride));
    MatMap ret_(ret.data() + ind_ * V, N, V, OuterStride(stride));

    ret_.noalias() = DERVALS * qh_;

    update_inds(inds, N);
  }
  ret /= dX(d);
}

void endpts(Matr ret, Matr qh, int d, int e, Matr ENDVALS, int ndim) {
  // ret[i] is value of qh at end e (0 or 1) of the dth axis
  // ret has shape (N^(ndim-1), V), qh has shape (N^ndim, V)
  // NOTE: qh must be contiguous

  int N = ENDVALS.cols();
  int V = qh.cols();
  int Nd_ = std::pow(N, ndim - 1);

  iVec inds = iVec::Zero(ndim - 1);
  int stride = std::pow(N, ndim - d - 1) * V;

  for (int ind = 0; ind < Nd_; ind++) {

    int ind_ = zero_index(inds, N, d);
    MatMap qh_(qh.data() + ind_ * V, N, V, OuterStride(stride));

    ret.row(ind) = ENDVALS.row(e) * qh_;

    update_inds(inds, N);
  }
}
