#ifndef CUSTOM_MAT_PROD_H
#define CUSTOM_MAT_PROD_H

#include "../types.h"
#include "Spectra/GenEigsSolver.h"
#include "eigen3/Core"

class CustomMatProd {
private:
  typedef Eigen::Index Index;
  typedef Eigen::Map<const Vec> MapConstVec;
  typedef Eigen::Map<Vec> MapVec;
  typedef const Eigen::Ref<const Mat> ConstGenericMatrix;

  ConstGenericMatrix m_mat;

public:
  CustomMatProd(ConstGenericMatrix &mat) : m_mat(mat) {}

  Index rows() const { return m_mat.rows(); }
  Index cols() const { return m_mat.cols(); }

  void perform_op(const double *x_in, double *y_out) const {
    MapConstVec x(x_in, m_mat.cols());
    MapVec y(y_out, m_mat.rows());
    y.noalias() = m_mat * x;
  }
};

typedef Spectra::GenEigsSolver<double, Spectra::LARGEST_MAGN, CustomMatProd>
    SpectraEigSolver;

#endif // CUSTOM_MAT_PROD_H
