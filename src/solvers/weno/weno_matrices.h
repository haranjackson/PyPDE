#ifndef WENO_MATRICES_H
#define WENO_MATRICES_H

#include <vector>

#include "../../scipy/math/polynomials.h"
#include "../../types.h"

std::vector<Mat> coefficient_matrices(const std::vector<poly> &basis, int FN2,
                                      int CN2);

Mat oscillation_indicator(const std::vector<poly> &basis);

#endif // WENO_MATRICES_H
