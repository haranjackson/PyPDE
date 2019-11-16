// Copyright (C) 2016-2019 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CUSTOM_MAT_PROD_H
#define CUSTOM_MAT_PROD_H

#include "Spectra/GenEigsSolver.h"
#include "eigen3/Core"
#include "types.h"

class CustomMatProd {
private:
  typedef Eigen::Index Index;
  typedef Eigen::Map<const Vec> MapConstVec;
  typedef Eigen::Map<Vec> MapVec;
  typedef const Eigen::Ref<const Mat> ConstGenericMatrix;

  ConstGenericMatrix m_mat;

public:
  ///
  /// Constructor to create the matrix operation object.
  ///
  /// \param mat An **Eigen** matrix object, whose type can be
  /// `Eigen::Matrix<double, ...>` (e.g. `Eigen::MatrixXd` and
  /// `Eigen::MatrixXf`), or its mapped version
  /// (e.g. `Eigen::Map<Eigen::MatrixXd>`).
  ///
  CustomMatProd(ConstGenericMatrix &mat) : m_mat(mat) {}

  ///
  /// Return the number of rows of the underlying matrix.
  ///
  Index rows() const { return m_mat.rows(); }
  ///
  /// Return the number of columns of the underlying matrix.
  ///
  Index cols() const { return m_mat.cols(); }

  ///
  /// Perform the matrix-vector multiplication operation \f$y=Ax\f$.
  ///
  /// \param x_in  Pointer to the \f$x\f$ vector.
  /// \param y_out Pointer to the \f$y\f$ vector.
  ///
  // y_out = A * x_in
  void perform_op(const double *x_in, double *y_out) const {
    MapConstVec x(x_in, m_mat.cols());
    MapVec y(y_out, m_mat.rows());
    y.noalias() = m_mat * x;
  }
};

typedef Spectra::GenEigsSolver<double, Spectra::LARGEST_MAGN, CustomMatProd>
    SpectraEigSolver;

#endif // CUSTOM_MAT_PROD_H
