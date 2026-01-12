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

#ifndef LAGHOS_ADAPTIVE_QUAD_HPP
#define LAGHOS_ADAPTIVE_QUAD_HPP

#include <algorithm>
#include <cmath>

///
/// Implements the 21-point adaptive Gauss-Kronrod quadrature method
///
template <class Fun, class Err> struct gk21 {
  Fun fun;
  Err err_fun;
  using res_type = decltype(fun(0.0));
  constexpr static size_t gl_points() { return 10; }
  constexpr static size_t gk_points() { return 11; }

private:
  res_type integrate_recurse(double lower, double upper, size_t curr_depth,
                             size_t max_depth = 20) const {
    static constexpr double data[] = {
        // gl_abscissa
        -1.488743389816312108848260011297200e-01,
        -4.333953941292471907992659431657842e-01,
        -6.794095682990244062343273651148736e-01,
        -8.650633666889845107320966884234930e-01,
        -9.739065285171717200779640120844521e-01,
        1.488743389816312108848260011297200e-01,
        4.333953941292471907992659431657842e-01,
        6.794095682990244062343273651148736e-01,
        8.650633666889845107320966884234930e-01,
        9.739065285171717200779640120844521e-01,
        // gl_weights
        2.955242247147528701738929946513383e-01,
        2.692667193099963550912269215694694e-01,
        2.190863625159820439955349342281632e-01,
        1.494513491505805931457763396576973e-01,
        6.667134430868813759356880989333179e-02,
        2.955242247147528701738929946513383e-01,
        2.692667193099963550912269215694694e-01,
        2.190863625159820439955349342281632e-01,
        1.494513491505805931457763396576973e-01,
        6.667134430868813759356880989333179e-02,
        // glk_weights
        1.477391049013384913748415159720680e-01,
        1.347092173114733259280540017717068e-01,
        1.093871588022976418992105903258050e-01,
        7.503967481091995276704314091619001e-02,
        3.255816230796472747881897245938976e-02,
        1.477391049013384913748415159720680e-01,
        1.347092173114733259280540017717068e-01,
        1.093871588022976418992105903258050e-01,
        7.503967481091995276704314091619001e-02,
        3.255816230796472747881897245938976e-02,
        // gk_abscissa
        0.000000000000000000000000000000000e00,
        -2.943928627014601981311266031038656e-01,
        -5.627571346686046833390000992726941e-01,
        -7.808177265864168970637175783450424e-01,
        -9.301574913557082260012071800595083e-01,
        -9.956571630258080807355272806890028e-01,
        2.943928627014601981311266031038656e-01,
        5.627571346686046833390000992726941e-01,
        7.808177265864168970637175783450424e-01,
        9.301574913557082260012071800595083e-01,
        9.956571630258080807355272806890028e-01,
        // gk_weights
        1.494455540029169056649364683898212e-01,
        1.427759385770600807970942731387171e-01,
        1.234919762620658510779581098310742e-01,
        9.312545458369760553506546508336634e-02,
        5.475589657435199603138130024458018e-02,
        1.169463886737187427806439606219205e-02,
        1.427759385770600807970942731387171e-01,
        1.234919762620658510779581098310742e-01,
        9.312545458369760553506546508336634e-02,
        5.475589657435199603138130024458018e-02,
        1.169463886737187427806439606219205e-02,
    };
    // TODO: where to copy gk21_base data to scratch memory?
    res_type gl_sum = 0;
    res_type gk_sum = 0;
    double jac = (upper - lower) * 0.5;
    for (int i = 0; i < gl_points(); ++i) {
      res_type f_eval = fun((data[i] + 1) * jac + lower);
      gl_sum += f_eval * data[gl_points() + i];
      gk_sum += f_eval * data[2 * gl_points() + i];
    }
    for (int i = 0; i < gk_points(); ++i) {
      res_type f_eval = fun((data[3 * gl_points() + i] + 1) * jac + lower);
      gk_sum += f_eval * data[3 * gl_points() + gk_points() + i];
    }
    gk_sum *= jac;
    gl_sum *= jac;
    if (curr_depth < max_depth && !err_fun(gk_sum, gl_sum)) {
      gk_sum = integrate_recurse(lower, lower + jac, curr_depth + 1, max_depth);
      gk_sum +=
          integrate_recurse(lower + jac, upper, curr_depth + 1, max_depth);
    }

    return gk_sum;
  }

public:
  res_type integrate(double lower, double upper, size_t start_segs = 1,
                     size_t max_depth = 20) const {
    double dx = (upper - lower) / start_segs;
    res_type res = 0;
    double curr = lower;
    for (size_t i = 0; i < start_segs; ++i) {
      double next = lower + (i + 1) * dx;
      res += integrate_recurse(curr, next, 1, max_depth);
      curr = next;
    }
    return res;
  }
};

template <class F, class E>
auto gk21_integrate(F &&f, E &&e, double lower, double upper,
                    size_t start_segs = 1, size_t max_depth = 20) {
  gk21<F, E> integrator{f, e};
  return integrator.integrate(lower, upper, start_segs, max_depth);
}

struct scalar_error_functor {
  double eps_abs;
  double eps_rel;
  template <class T> bool operator()(const T &ho, const T &lo) const {
    if (!std::isfinite(ho)) {
      return true;
    }
    double delta = std::fabs(ho - lo);
    if (delta < eps_abs) {
      return true;
    }
    double denom = std::max(std::fabs(ho), std::fabs(lo));
    if (delta < eps_rel * denom) {
      return true;
    }
    return false;
  }
};

#endif
