#include "../types.h"

void update_inds(iVecr inds, iVecr bounds) {
  // Given inds=(i1, i2, ...) and bounds=(n1, n2, ...), increments inds in such
  // a way that the index in the flattened grid is incremented by 1
  for (int i = inds.size() - 1; i >= 0; i--) {

    inds(i) += 1;

    if (inds(i) == bounds(i)) {
      inds(i) = 0;
    } else {
      break;
    }
  }
}

void update_inds(iVecr inds, int N) {
  // Given inds=(i1, i2, ...) and 0 <= i < N, increments inds in such a
  // way that the index in the flattened grid is incremented by 1

  for (int i = inds.size() - 1; i >= 0; i--) {

    inds(i) += 1;

    if (inds(i) == N) {
      inds(i) = 0;
    } else {
      break;
    }
  }
}

int index(iVecr inds, iVecr bounds, int offset) {
  // Given inds=(i1, i2, ...) and bounds=(n1, n2, ...), returns the index in the
  // flattened grid of (i1+offset, i2+offset, ...)
  int ret = inds(0) + offset;

  for (int i = 1; i < inds.size(); i++) {
    ret *= bounds(i);
    ret += inds(i) + offset;
  }

  return ret;
}

int index(iVecr inds, int N) {
  // Given inds=(i1, i2, ...) and 0 <= i < N, returns the index in the
  // flattened grid
  int ret = inds(0);

  for (int i = 1; i < inds.size(); i++) {
    ret *= N;
    ret += inds(i);
  }

  return ret;
}