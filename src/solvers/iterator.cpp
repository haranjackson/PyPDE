#include "../grid/boundaries.h"
#include "../types.h"
#include "ThreadPool.h"
#include "dg/dg.h"
#include "fv/fv.h"
#include "stepper.h"
#include "weno/weno.h"

#include <algorithm>
#include <iostream>
#include <thread>

iVec nX_extend(iVecr nX, int ext) {
  iVec ret = nX;
  ret.array() += 2 * ext;
  return ret;
}

void copy_parts(Matr arr, std::vector<Mat> &arrParts, int overlap,
                bool reverse) {

  int rowStart = 0;
  int cols = arrParts[0].cols();

  for (int i = 0; i < arrParts.size(); i++) {

    int rows = arrParts[i].rows();

    if (reverse)
      arrParts[i] = arr.block(rowStart, 0, rows, cols);
    else
      arr.block(rowStart, 0, rows, cols) = arrParts[i];

    rowStart += rows - overlap;
  }
}

void iterator(void (*F)(double *, double *, double *, int),
              void (*B)(double *, double *, int), void (*S)(double *, double *),
              Matr u, double tf, iVecr nX, aVecr dX, double CFL,
              iVecr boundaryTypes, bool STIFF, int FLUX, int N, int ndt,
              bool secondOrder, Matr ret, int nThreads) {

  int V = u.size() / nX.prod();

  iVec nXb = nX_extend(nX, N);

  WenoSolver wenoSolver(nXb, N, V);

  if (nThreads < 1)
    nThreads = std::min(1, int(std::thread::hardware_concurrency() - 1));

  std::cout << "Using " << nThreads << " threads\n";

  std::vector<TimeStepper> timeSteppers;
  std::vector<DGSolver> dgSolvers;
  std::vector<FVSolver> fvSolvers;

  std::vector<Mat> wParts;
  std::vector<Mat> uParts;

  iVec nXw = nX_extend(nX, 1);
  int ndim = nX.size();

  int uRow = nX.tail(ndim - 1).prod();
  int wRow = nXw.tail(ndim - 1).prod() * std::pow(N, ndim);

  for (int i = 0; i < nThreads; i++) {

    int start = i * nX(0) / nThreads;
    int finish = (i + 1) * nX(0) / nThreads;

    int uWidth = finish - start;

    uParts.emplace_back(uWidth * uRow, V);
    wParts.emplace_back((uWidth + 2) * wRow, V);

    iVec nXpart = nX;
    nXpart(0) = uWidth;

    timeSteppers.emplace_back(F, B, dX, N, V, CFL, tf, secondOrder);
    dgSolvers.emplace_back(F, B, S, dX, STIFF, N, V);
    fvSolvers.emplace_back(F, B, S, nXpart, dX, FLUX, N, V, secondOrder);
  }

  Mat uprev = u;

  ThreadPool pool(nThreads);

  std::vector<std::future<double>> dtResults(nThreads);
  std::vector<std::future<void>> fvResults(nThreads);

  double t = 0.;
  long count = 0;
  int pushCount = 0;

  copy_parts(u, uParts, 0, true);

  while (t < tf) {

    Mat ub = boundaries(u, nX, boundaryTypes, N);

    Mat w = wenoSolver.reconstruction(ub);

    copy_parts(w, wParts, 2 * wRow, true);

    for (int i = 0; i < nThreads; ++i) {
      auto f = [i, t, count, &timeSteppers, &wParts] {
        return timeSteppers[i].step(wParts[i], t, count);
      };
      dtResults[i] = pool.enqueue(f);
    }

    double dt = INF;
    for (auto &&result : dtResults)
      dt = std::min(result.get(), dt);

    for (int i = 0; i < nThreads; ++i) {
      auto f = [i, dt, &dgSolvers, &fvSolvers, &wParts, &uParts] {
        Mat qh = dgSolvers[i].predictor(wParts[i], dt);
        fvSolvers[i].apply(uParts[i], qh, dt);
      };
      fvResults[i] = pool.enqueue(f);
    }

    for (auto &&result : fvResults)
      result.wait();

    copy_parts(u, uParts, 0, false);

    t += dt;
    count += 1;

    std::cout << "t = " << t << "\n";

    if (t >= double(pushCount + 1) / double(ndt) * tf) {
      ret.row(pushCount) = VecMap(u.data(), u.size());
      pushCount += 1;
    }

    if (u.array().isNaN().any()) {
      std::cout << "NaNs found";
      ret.row(pushCount) = VecMap(uprev.data(), uprev.size());
      ret.row(pushCount + 1) = VecMap(u.data(), u.size());
    }

    uprev = u;
  }

  ret.row(ndt - 1) = VecMap(u.data(), u.size());
}
