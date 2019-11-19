#ifndef DG_MATRICES_H
#define DG_MATRICES_H

#include "../../scipy/math/polynomials.h"
#include "../../types.h"

Mat kron(std::vector<Mat> &mats);

Mat end_value_products(const std::vector<poly> &basis);

Mat derivative_products(const std::vector<poly> &basis, Vecr NODES, Vecr WGHTS);

#endif // DG_MATRICES_H
