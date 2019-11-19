#ifndef BASIS_H
#define BASIS_H

#include <vector>

#include "../scipy/math/polynomials.h"
#include "../types.h"

Vec scaled_nodes(int N);

Vec scaled_weights(int N);

std::vector<poly> basis_polys(int N);

poly lagrange(Vecr x, int i);

Mat end_values(const std::vector<poly> &basis);

Mat derivative_values(const std::vector<poly> &basis, Vecr NODES);

#endif // BASIS_H
