#include <cmath>

#include "../../types.h"
#include "lgmres.h"

Vec lgmres(VecFunc matvec, VecFunc psolve, Vecr b, Vec x,
           std::vector<Vec> &outer_v, const double tol, const int maxiter,
           const int inner_m, const int outer_k) {

  int n = b.size();

  double b_norm = b.norm();
  if (b_norm == 0.)
    b_norm = 1.;

  Vec r_outer(n);
  Vec vs0(n);
  Vec mvz(n);
  Vec z(n);
  DecQR QR;

  for (int k_outer = 0; k_outer < maxiter; k_outer++) {
    r_outer = matvec(x) - b;

    double r_norm = r_outer.norm();
    if (r_norm <= tol * b_norm || r_norm <= tol)
      break;

    vs0 = -psolve(r_outer);
    double inner_res_0 = vs0.norm();

    vs0 /= inner_res_0;
    std::vector<Vec> vs = {vs0};
    std::vector<Vec> ws;

    int ind = 1 + inner_m + outer_v.size();

    Mat A = Mat(ind, ind);
    Mat Q = Mat::Zero(ind, ind);
    Mat R = Mat::Zero(ind, ind - 1);
    Q(0, 0) = 1.;

    int j = 1;
    for (; j < ind; j++) {

      if (j < outer_v.size() + 1)
        z = outer_v[j - 1];
      else if (j == outer_v.size() + 1)
        z = vs0;
      else
        z = vs.back();

      mvz = matvec(z);
      Vec v_new = psolve(mvz);
      double v_new_norm = v_new.norm();

      Vec hcur = Vec::Zero(j + 1);
      for (int i = 0; i < vs.size(); i++) {
        double alpha = vs[i].dot(v_new);
        hcur(i) = alpha;
        v_new -= alpha * vs[i];
      }
      hcur(j) = v_new.norm();

      v_new /= hcur(j);
      vs.push_back(v_new);
      ws.push_back(z);

      Q(j, j) = 1.;

      A.topLeftCorner(j + 1, j - 1) =
          Q.topLeftCorner(j + 1, j + 1) * R.topLeftCorner(j + 1, j - 1);
      A.block(0, j - 1, j + 1, 1) = hcur;
      QR.compute(A.topLeftCorner(j + 1, j));

      Q.topLeftCorner(j + 1, j + 1) = QR.householderQ();
      R.topLeftCorner(j + 1, j) = QR.matrixQR().triangularView<Eigen::Upper>();

      double inner_res = std::abs(Q(0, j)) * inner_res_0;

      if ((inner_res <= tol * inner_res_0) || (hcur(j) <= mEPS * v_new_norm))
        break;
    }
    if (j == ind)
      j -= 1;

    Vec y = R.topLeftCorner(j, j).colPivHouseholderQr().solve(
        Q.topLeftCorner(1, j).transpose());
    y *= inner_res_0;

    Vec dx = y(0) * ws[0];
    for (int i = 1; i < y.size(); i++)
      dx += y(i) * ws[i];

    double nx = dx.norm();
    if (nx > 0.)
      outer_v.push_back(dx / nx);

    while (outer_v.size() > outer_k)
      outer_v.erase(outer_v.begin());
    x += dx;
  }
  return x;
}

Vec lgmres_wrapper(Matr A, Vecr b, Vecr x0, Matr M, const double tol,
                   const int maxiter, const int inner_m, const int outer_k,
                   std::vector<Vec> &outer_v) {
  System system;
  system.set_A(A);

  if (M.rows() == 0 && M.cols() == 0)
    system.set_M(Mat::Identity(A.cols(), A.rows()));
  else
    system.set_M(M);

  using std::placeholders::_1;
  std::function<Vec(Vecr)> matvec = std::bind(&System::call, system, _1);
  std::function<Vec(Vecr)> psolve = std::bind(&System::prec, system, _1);

  if (x0.rows() == 0)
    return lgmres(matvec, psolve, b, Vec::Zero(b.rows()), outer_v, tol, maxiter,
                  inner_m, outer_k);
  else
    return lgmres(matvec, psolve, b, x0, outer_v, tol, maxiter, inner_m,
                  outer_k);
}
