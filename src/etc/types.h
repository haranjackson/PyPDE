#ifndef TYPES_H
#define TYPES_H

//#define EIGEN_USE_MKL_ALL
//#define MKL_DIRECT_CALL

//#define EIGEN_NO_MALLOC

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "eigen3/Eigen"
#include "eigen3/StdVector"

#include <limits>

const double mEPS = 2.2204460492503131e-16;
const double INF = std::numeric_limits<double>::max();

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    aMat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> iVec;
typedef Eigen::Array<double, Eigen::Dynamic, 1> aVec;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> cVec;

typedef Eigen::Ref<Mat> Matr;
typedef Eigen::Ref<aMat> aMatr;
typedef Eigen::Ref<Vec> Vecr;
typedef Eigen::Ref<iVec> iVecr;
typedef Eigen::Ref<aVec> aVecr;
typedef Eigen::Ref<cVec> cVecr;

typedef const Eigen::Ref<const Mat> Matr_c;
typedef const Eigen::Ref<const Vec> Vecr_c;
typedef const Eigen::Ref<const iVec> iVecr_c;

typedef Eigen::OuterStride<Eigen::Dynamic> OuterStride;
typedef Eigen::Map<Mat, 0, Eigen::OuterStride<Eigen::Dynamic> > MatMap;
typedef Eigen::Map<Vec, 0, Eigen::InnerStride<1> > VecMap;
typedef Eigen::Map<aVec, 0, Eigen::InnerStride<1> > aVecMap;
typedef Eigen::Map<iVec, 0, Eigen::InnerStride<1> > iVecMap;

typedef Eigen::HouseholderQR<Mat> DecQR;
typedef Eigen::ColPivHouseholderQR<Mat> Dec;

typedef std::function<Vec(Vecr)> VecFunc;

template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

#endif
