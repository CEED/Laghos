// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef LAGHOS_BISECT_HPP
#define LAGHOS_BISECT_HPP

#include <cmath>
#include <stdexcept>

#include <iostream>

/// Bisection root finder
template <class Fun> double bisection(Fun &&fun, double lower, double upper) {
  double lv = fun(lower);
  constexpr double tol = 1e-20;
  if (std::fabs(lv) < tol) {
    return lower;
  }
  double rv = fun(upper);
  if (std::fabs(rv) < tol) {
    return upper;
  }
  if (std::copysign(1., lv) * std::copysign(1., rv) > 0) {
    throw std::runtime_error("bisection: no sign change");
  }
  auto dx_init = upper - lower;
  auto dx_last = dx_init;
  while (true) {
    double mid = 0.5 * (lower + upper);
    auto dx = mid - lower;
    double mv = fun(mid);
    if (dx < dx_init * 1e-16 || dx >= dx_last) {
      if (fabs(mv) < fabs(lv)) {
        if (fabs(mv) < fabs(rv)) {
          return mid;
        } else if (fabs(rv) < fabs(lv)) {
          return upper;
        } else {
          return lower;
        }
      } else if (fabs(rv) < fabs(lv)) {
        return upper;
      } else {
        return lower;
      }
    }
    if (std::fabs(mv) < tol) {
      return mid;
    }
    if (std::copysign(1., lv) != std::copysign(1., mv)) {
      upper = mid;
      rv = mv;
    } else if (std::copysign(1., rv) != std::copysign(1., mv)) {
      lower = mid;
      lv = mv;
    } else {
      throw std::runtime_error("bisection: no sign change");
    }
    dx_last = dx;
  }
}

#endif
