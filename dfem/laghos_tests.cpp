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
//
// Non-regression tests of the dFEM Laghos explicit and implicit paths, with
// the Catch2 of MFEM's tests/unit/catch.hpp:
//
//   laghos_tests                  runs all the tests
//   laghos_tests exp_run_sdv_2d_q21_rk3_av7_dtn runs one test, as done by CTest
//   laghos_tests "[tgv]"          runs the tests with a given tag
//   laghos_tests "[exp]"          runs the explicit tests only, ctest -L exp
//   laghos_tests "[imp]"          runs the implicit tests only, ctest -L imp
//   laghos_tests "[chk]"          runs the README checks only, ctest -L chk
//   laghos_tests -v ...           also shows the raw Laghos output
//   laghos_tests --device debug   runs on a Device, created once per process
//   laghos_tests --device cuda    runs on a GPU, one test per process in CTest
//   LAGHOS_TESTS_GPU=2 laghos_tests --device cuda  runs from GPU 2
//   laghos_tests --device cuda --gpus  prints the GPU count, for the makefile
//   laghos_tests --qf global      runs with global q-functions, ctest -L glb
//   laghos_tests -h               Catch2 help

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "laghos_tests.hpp"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

using mfem::real_t;

#ifndef LAGHOS_DATA_DIR
#define LAGHOS_DATA_DIR "../data/"
#endif

static bool verbose = false; // -v: keep the raw Laghos output
static bool gpus = false; // --gpus: print the GPU count only
static std::string device = "cpu"; // --device: Device configuration string
static std::string qf = "local"; // --qf: q-function backend, local or global
static int device_id = 0; // GPU of the rank, see main
static struct { int passed = 0, total = 0; } summary; // final report line

///////////////////////////////////////////////////////////////////////////////
template <class T>
std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 10.0*std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0) { return neg < eps; }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

///////////////////////////////////////////////////////////////////////////////
struct Test
{
   static constexpr const char *binary = "laghos";
   static constexpr const char *common =
      "-no-glvis -no-paraview -no-save -vs 1000";
   const std::string name, desc, mesh, options;
   const int steps;
   const real_t dt, energy, x_norm, v_norm, e_norm, tolerance;
   Test(const char *name, const char *desc, const char *mesh,
        const char *options,
        const int steps, const real_t dt, const real_t energy,
        const real_t x_norm, const real_t v_norm, const real_t e_norm,
        const real_t tolerance = 1e-10):
      name(name), desc(desc), mesh(mesh),
      options(std::string(options) + " " + common),
      steps(steps), dt(dt), energy(energy),
      x_norm(x_norm), v_norm(v_norm), e_norm(e_norm), tolerance(tolerance) {}
   std::string Command() const
   {
      return std::string(binary) + " -m " + LAGHOS_DATA_DIR + mesh + " " +
             options + " -d " + device + " -dev " + std::to_string(device_id) +
             " -qf " + qf;
   }
};

///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      "exp_run_sdv_2d_q21_rk3_av7_dtn", "explicit RK3SSP, Q2-Q1, av-type 7, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.05",
      23,
      1.5436933912796560e-04, 2.5004931952471232e-01,
      1.4101829631350940e+01, 4.4195254279849605e+00, 4.2208594277324963e+01
   },
   {
      "exp_run_sdv_2d_q21_rk2_av7_dtn", "explicit RK2, Q2-Q1, av-type 7, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 2 -cfl 0.5 -av -av-type 7 -tf 0.05",
      6,
      6.2443204309436178e-03, 2.4980489638517073e-01,
      7.5779033060146155e+00, 1.5104070411987085e+00, 1.4249329026179177e+01
   },
   {
      "exp_run_sdv_3d_q21_rk3_av7_dtn", "explicit RK3SSP, Q2-Q1, av-type 7, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.02",
      6,
      9.7893419572273734e-06, 1.2501143644784798e-01,
      2.7832196671533371e+01, 3.3771156618999734e+00, 5.7714521901242946e+01
   },
   // Stock-comparable Sedov: stock Laghos tensor viscosity (-av-type 1) and
   // time step estimate (-dt-min), same steps, dt, ΔE and |e| as
   //   laghos -dim 2 -nx 1 -ny 1 -rs 3 -p 1 -ok 2 -ot 1 -s 3 -cfl 0.5 -tf 0.05 -fa
   //   laghos -dim 3 -nx 2 -ny 2 -nz 2 -rs 1 -p 1 -ok 2 -ot 1 -s 3 -cfl 0.5 -tf 0.02 -fa
   {
      "exp_run_sdv_2d_q21_rk3_av1_dtm", "explicit RK3SSP, Q2-Q1, stock av, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.05",
      22, 1.5244822594611736e-03, 2.5005546874349194e-01, 1.4102287121585931e+01,
      4.5117390396546329e+00, 4.0702975696596390e+01
   },
   {
      "exp_run_sdv_3d_q21_rk3_av1_dtm", "explicit RK3SSP, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.02"
      " -cgt 1e-10",
      4, 4.3725299546347671e-03, 1.2503911283965757e-01, 2.7832229285175924e+01,
      3.4262677649627276e+00, 5.6552559708157922e+01
   },
   {
      "exp_run_tgv_2d_q32_rk4_av0_dtn", "explicit RK4, Q3-Q2, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 3 -oe 2 -s 4 -cfl 0.5 -tf 0.05",
      3,
      5.3791888466399942e-03, 1.7499984739484764e+00,
      1.0847749476402059e+01, 9.1602573868358732e+00, 1.8694024191662098e+01
   },
   {
      "exp_run_tgv_3d_q21_rk4_av0_dtn", "explicit RK4, Q2-Q1, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05",
      7,
      1.9746163022951384e-03, 1.4993325901494498e+02,
      1.1859990900590070e+01, 6.2046596711615782e+00, 1.1985122282081534e+03
   },
   {
      "exp_run_tgv_3d_q21_rk4_av7_dtn", "explicit RK4, Q2-Q1, av-type 7, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 7 -tf 0.05",
      18,
      1.4960934056207102e-03, 1.4993325988669309e+02,
      1.1858399916887008e+01, 9.9668537477810759e-01, 1.1994457713114568e+03
   },
   // Checks: smaller runs of the README 'Verification of Results' cases, with
   // stock defaults (Q2-Q1, RK4, cfl 0.5), compared with (data/ meshes, -pa)
   //   laghos -p 0 -m square01_quad.mesh -rs 1 -tf 0.05   (README 1)
   //   laghos -p 0 -m cube01_hex.mesh -rs 1 -tf 0.01      (README 2)
   //   laghos -p 1 -m square01_quad.mesh -rs 2 -tf 0.05   (README 3)
   //   laghos -p 1 -m cube01_hex.mesh -rs 1 -tf 0.02      (README 4, -E0 1)
   // dFEM Laghos starts from a 1x1 square, one more -rs than the 2x2 mesh
   {
      "exp_chk_tgv_2d_q21_rk4_av0_dtn",
      "README 1, explicit RK4, Q2-Q1, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05",
      2, 1.7273414632041139e-02, 1.7496138442784803e+00, 7.5755635427647361e+00,
      6.3210512132666858e+00, 1.2508147843915047e+01
   },
   {
      "exp_chk_tgv_3d_q21_rk4_av0_dtn",
      "README 2, explicit RK4, Q2-Q1, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 1 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.01",
      3, 1.8178734530672098e-03, 1.4993721050652641e+02, 2.7831009958537454e+01,
      1.4156587512698188e+01, 3.3898742905766835e+03
   },
   {
      "exp_chk_sdv_2d_q21_rk4_av1_dtm",
      "README 3, explicit RK4, Q2-Q1, stock av, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.05",
      21, 1.8984422273346600e-03, 2.5000401709103320e-01, 1.4102285371332647e+01,
      4.5108287429679663e+00, 4.0693374548498461e+01
   },
   {
      "exp_chk_sdv_3d_q21_rk4_av1_dtm",
      "README 4, explicit RK4, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.02"
      " -cgt 1e-10",
      4, 4.3725299546347671e-03, 1.2500220220201469e-01, 2.7832227632577982e+01,
      3.4200678875411525e+00, 5.6528006524777794e+01
   },
   // More 3D: lowest order Q1-Q0, stock av on a smooth flow, and the triple
   // point, the only problem reading its -m mesh (box with two materials),
   // stock-comparable runs as
   //   laghos -p 0 -dim 3 -m cube01_hex.mesh -rs 1 -ok 1 -ot 0 -s 4 -tf 0.05
   //   laghos -p 1 -dim 3 -m cube01_hex.mesh -rs 1 -ok 1 -ot 0 -s 3 -tf 0.1
   //   laghos -p 0 -dim 3 -m cube01_hex.mesh -rs 0 -ok 2 -ot 1 -s 4 -iv -tf 0.05
   //   laghos -p 3 -dim 3 -m box01_hex.mesh -rs 0 -ok 2 -ot 1 -s 4 -tf 0.2
   {
      "exp_run_tgv_3d_q10_rk4_av0_dtn", "explicit RK4, Q1-Q0, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 1 -ov 1 -oe 0 -s 4 -cfl 0.5 -tf 0.05",
      7, 6.6060646756573873e-04, 1.4990434628565481e+02, 1.1859894505936010e+01,
      5.5879520246439682e+00, 1.1985981660785569e+03
   },
   {
      "exp_run_sdv_3d_q10_rk3_av1_dtm", "explicit RK3SSP, Q1-Q0, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 1 -oe 0 -s 3 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.1",
      6, 4.5106785655475044e-03, 1.2526730824116419e-01, 1.1868845859494494e+01,
      1.9848019446683176e+00, 6.8841771094494320e+00
   },
   {
      "exp_run_tgv_3d_q21_rk4_av1_dtm", "explicit RK4, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.05",
      14, 1.1438157810123699e-03, 1.4993326044731037e+02, 1.1858537217399940e+01,
      1.7347628571356015e+00, 1.1993939640687433e+03
   },
   {
      "exp_run_tpt_3d_q21_rk4_av1_dtm", "explicit RK4, Q2-Q1, stock av, 3D",
      "box01_hex.mesh",
      "-p 3 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -dt-min -tf 0.2",
      5, 3.2015046545843445e-02, 3.0150228406430880e+01, 7.0909362959399473e+01,
      2.1011239969290898e+00, 1.6921507332503591e+01
   },
   {
      "exp_run_tpt_3d_q21_rk4_av7_dtn", "explicit RK4, Q2-Q1, av-type 7, 3D",
      "box01_hex.mesh",
      "-p 3 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 7 -tf 0.2",
      10, 1.0248581728371387e-02, 3.0150198118370763e+01, 7.0902482097110010e+01,
      1.3479830805551927e+00, 1.6939266983223721e+01
   },
   // Implicit ImplicitMidpoint (-s 12) and BackwardEuler (-s 11), Newton with
   // GMRES and block-diagonal AMG (-pt 0), references of the
   // pre-dfem-dev MFEM (01-dfem-laghos-dev-av 7cc9281ce8, laghos-dfem 157bcf1)
   {
      "imp_run_tgv_2d_q21_mid_av0_dtn", "implicit midpoint, Q2-Q1, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 2 -oe 1 -s 12 -cfl 8 -tf 0.25 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      1, 2.5000000000000000e-01, 1.7496144709791834e+00, 7.6158393763847849e+00,
      6.1657639728600806e+00, 1.2801715728172615e+01, 1e-8
   },
   {
      "imp_run_tgv_2d_q21_bwe_av0_dtn",
      "implicit backward Euler, Q2-Q1, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 2 -oe 1 -s 11 -cfl 8 -tf 0.25 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      1, 2.5000000000000000e-01, 1.7144424302407082e+00, 7.5355450090088745e+00,
      5.5950119610093045e+00, 1.3076511553426874e+01, 1e-8
   },
   {
      "imp_run_sdv_2d_q21_mid_av0_dtn", "implicit midpoint, Q2-Q1, no av, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 12 -cfl 8 -tf 0.1 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      2, 4.7799374999999998e-02, 2.4999999998372771e-01, 7.5895841452573265e+00,
      3.3576691897422939e+00, 1.0426342267073755e+01, 1e-8
   },
   {
      "imp_run_sdv_2d_q21_mid_av7_dtn",
      "implicit midpoint, Q2-Q1, av-type 7, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 12 -cfl 8 -av -av-type 7 -tf 0.1 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      10, 7.5072974222040595e-03, 2.4999999999994080e-01, 7.5866802435482432e+00,
      1.8988891018276433e+00, 1.2690442177254132e+01, 1e-8
   },
   {
      "imp_run_sdv_2d_q21_mid_av1_dtn",
      "implicit midpoint, Q2-Q1, av-type 1, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 12 -cfl 8 -av -av-type 1 -tf 0.1 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      14, 3.4735878931168307e-03, 2.4999999999960013e-01, 7.5873290046513846e+00,
      1.9166132239675209e+00, 1.2236668098815036e+01, 1e-8
   },
   {
      "imp_run_tgv_3d_q21_mid_av0_dtn", "implicit midpoint, Q2-Q1, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 12 -cfl 8 -tf 0.1 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      1, 1.0000000000000001e-01, 1.4993325955324906e+02, 1.1867410366035584e+01,
      6.3036438007152293e+00, 1.1986499229245242e+03, 1e-8
   },
   {
      "imp_run_sdv_3d_q21_mid_av7_dtn",
      "implicit midpoint, Q2-Q1, av-type 7, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 12 -cfl 8 -av -av-type 7 -tf 0.02 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0",
      4, 7.6537470312500536e-04, 1.2499999999868094e-01, 2.7832191726334166e+01,
      3.3944688370441605e+00, 5.7726749810098475e+01, 1e-8
   },
};

///////////////////////////////////////////////////////////////////////////////
static const Test &Find(const std::string &name)
{
   for (auto &run : runs) { if (run.name == name) { return run; } }
   throw std::runtime_error("Unknown test: " + name);
}

///////////////////////////////////////////////////////////////////////////////
static void LaghosTest(const std::string &name)
{
   const Test &test = Find(name);

   std::vector<std::string> tokens;
   std::istringstream iss(test.Command());
   for (std::string token; iss >> token;) { tokens.push_back(token); }
   std::vector<char*> args;
   for (auto &token : tokens) { args.push_back(&token[0]); }
   args.push_back(nullptr);

   // Header before the run, naming the test case of a crash
   const bool root = mfem::Mpi::Root();
   if (root)
   {
      report::Header(test.name, test.desc +
                     (qf == "local" ? "" : ", " + qf) +
                     (device == "cpu" ? "" : ", " + device));
   }

   // Quiet Laghos run, unless -v
   LaghosResults res;
   const bool out_enabled = mfem::out.IsEnabled();
   if (!verbose) { mfem::out.Disable(); }
   laghos_results = &res;
   const int ret = laghos(static_cast<int>(args.size()) - 1, args.data());
   laghos_results = nullptr;
   if (out_enabled) { mfem::out.Enable(); }

   if (root && ret != 0) { report::Row().Fail("laghos").Int("returned", ret); }
   REQUIRE(ret == 0);

   struct Check
   {
      const char *label; real_t value, ref; bool ok;
      real_t Rel() const
      {
         return std::abs(value - ref) /
                std::max(std::abs(ref), std::numeric_limits<real_t>::min());
      }
   };
   const Check checks[] =
   {
      {"dt", res.dt, test.dt, AlmostEq(res.dt, test.dt, test.tolerance)},
      {"E", res.energy_final, test.energy, AlmostEq(res.energy_final, test.energy, test.tolerance)},
      {"‖x‖", res.x_norm, test.x_norm, AlmostEq(res.x_norm, test.x_norm, test.tolerance)},
      {"‖v‖", res.v_norm, test.v_norm, AlmostEq(res.v_norm, test.v_norm, test.tolerance)},
      {"‖e‖", res.e_norm, test.e_norm, AlmostEq(res.e_norm, test.e_norm, test.tolerance)},
   };
   const bool steps_ok = res.steps == test.steps;
   bool ok = steps_ok;
   real_t rel_max = 0.0;
   for (const auto &c : checks)
   {
      ok = ok && c.ok;
      rel_max = std::isnan(c.Rel()) ? c.Rel() : std::max(rel_max, c.Rel());
   }
   ++summary.total;
   if (ok) { ++summary.passed; }

   if (root)
   {
      report::Row().Tag("setup").Int("el", res.ne, 4)
      .Int("H1", res.h1_dofs, 5).Int("L2", res.l2_dofs, 4).Num("dt₀", res.dt_init);

      // First, middle and last accepted time steps
      const int n = static_cast<int>(res.history.size());
      std::vector<int> picks = {0, n/2, n-1};
      picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
      for (const int k : picks)
      {
         if (k < 0 || k >= n) { continue; }
         const auto &h = res.history[k];
         report::Row().Tag("step " + std::to_string(h.ti))
         .Num("t", h.t).Num("dt", h.dt);
      }
      report::Row().Color(report::Yellow()).Tag("energy").Num("E₀", res.energy_init)
      .Num("E", res.energy_final)
      .Num("|ΔE|", std::abs(res.energy_final - res.energy_init), 2)
      .Int("repeats", res.repeats, 2);
      {
         report::Row row;
         row.Tag("check").Int("steps", res.steps, 3);
         for (const auto &c : checks) { row.Num(c.label, c.value); }
         row.Rel("rel", rel_max, test.tolerance).Mark(ok);
      }

      if (!ok)
      {
         if (!steps_ok)
         {
            report::Row().Fail("steps").Int("got", res.steps)
            .Int("ref", test.steps);
         }
         for (const auto &c : checks)
         {
            if (c.ok) { continue; }
            report::Row(16).Fail(c.label).Num("got", c.value)
            .Num("ref", c.ref).Rel("rel", c.Rel(), test.tolerance);
         }
         // Same layout as the runs[] reference values, to update them if needed
         std::ostringstream os;
         os << std::scientific << std::setprecision(16) << res.steps << ", "
            << res.dt << ", " << res.energy_final << ", " << res.x_norm << ", "
            << res.v_norm << ", " << res.e_norm;
         report::Row().Tag("ref").Text(os.str());
      }
   }

   CHECK(res.steps == test.steps);
   for (const auto &c : checks) { INFO(c.label); CHECK(c.ok); }
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE("exp_run_sdv_2d_q21_rk3_av7_dtn",
          "[dfem][exp][run][sdv][2d][q21][dtn]")
{ LaghosTest("exp_run_sdv_2d_q21_rk3_av7_dtn"); }

TEST_CASE("exp_run_sdv_2d_q21_rk2_av7_dtn",
          "[dfem][exp][run][sdv][2d][q21][dtn]")
{ LaghosTest("exp_run_sdv_2d_q21_rk2_av7_dtn"); }

TEST_CASE("exp_run_sdv_3d_q21_rk3_av7_dtn",
          "[dfem][exp][run][sdv][3d][q21][dtn]")
{ LaghosTest("exp_run_sdv_3d_q21_rk3_av7_dtn"); }

TEST_CASE("exp_run_sdv_2d_q21_rk3_av1_dtm",
          "[dfem][exp][run][sdv][2d][q21][dtm]")
{ LaghosTest("exp_run_sdv_2d_q21_rk3_av1_dtm"); }

TEST_CASE("exp_run_sdv_3d_q21_rk3_av1_dtm",
          "[dfem][exp][run][sdv][3d][q21][dtm]")
{ LaghosTest("exp_run_sdv_3d_q21_rk3_av1_dtm"); }

TEST_CASE("exp_run_tgv_2d_q32_rk4_av0_dtn",
          "[dfem][exp][run][tgv][2d][q32][dtn]")
{ LaghosTest("exp_run_tgv_2d_q32_rk4_av0_dtn"); }

TEST_CASE("exp_run_tgv_3d_q21_rk4_av0_dtn",
          "[dfem][exp][run][tgv][3d][q21][dtn]")
{ LaghosTest("exp_run_tgv_3d_q21_rk4_av0_dtn"); }

TEST_CASE("exp_run_tgv_3d_q21_rk4_av7_dtn",
          "[dfem][exp][run][tgv][3d][q21][dtn]")
{ LaghosTest("exp_run_tgv_3d_q21_rk4_av7_dtn"); }

TEST_CASE("exp_chk_tgv_2d_q21_rk4_av0_dtn",
          "[dfem][exp][chk][tgv][2d][q21][dtn]")
{ LaghosTest("exp_chk_tgv_2d_q21_rk4_av0_dtn"); }

TEST_CASE("exp_chk_tgv_3d_q21_rk4_av0_dtn",
          "[dfem][exp][chk][tgv][3d][q21][dtn]")
{ LaghosTest("exp_chk_tgv_3d_q21_rk4_av0_dtn"); }

TEST_CASE("exp_chk_sdv_2d_q21_rk4_av1_dtm",
          "[dfem][exp][chk][sdv][2d][q21][dtm]")
{ LaghosTest("exp_chk_sdv_2d_q21_rk4_av1_dtm"); }

TEST_CASE("exp_chk_sdv_3d_q21_rk4_av1_dtm",
          "[dfem][exp][chk][sdv][3d][q21][dtm]")
{ LaghosTest("exp_chk_sdv_3d_q21_rk4_av1_dtm"); }

TEST_CASE("exp_run_tgv_3d_q10_rk4_av0_dtn",
          "[dfem][exp][run][tgv][3d][q10][dtn]")
{ LaghosTest("exp_run_tgv_3d_q10_rk4_av0_dtn"); }

TEST_CASE("exp_run_sdv_3d_q10_rk3_av1_dtm",
          "[dfem][exp][run][sdv][3d][q10][dtm]")
{ LaghosTest("exp_run_sdv_3d_q10_rk3_av1_dtm"); }

TEST_CASE("exp_run_tgv_3d_q21_rk4_av1_dtm",
          "[dfem][exp][run][tgv][3d][q21][dtm]")
{ LaghosTest("exp_run_tgv_3d_q21_rk4_av1_dtm"); }

TEST_CASE("exp_run_tpt_3d_q21_rk4_av1_dtm",
          "[dfem][exp][run][tpt][3d][q21][dtm]")
{ LaghosTest("exp_run_tpt_3d_q21_rk4_av1_dtm"); }

TEST_CASE("exp_run_tpt_3d_q21_rk4_av7_dtn",
          "[dfem][exp][run][tpt][3d][q21][dtn]")
{ LaghosTest("exp_run_tpt_3d_q21_rk4_av7_dtn"); }

TEST_CASE("imp_run_tgv_2d_q21_mid_av0_dtn",
          "[dfem][imp][run][tgv][2d][q21][dtn]")
{ LaghosTest("imp_run_tgv_2d_q21_mid_av0_dtn"); }

TEST_CASE("imp_run_tgv_2d_q21_bwe_av0_dtn",
          "[dfem][imp][run][tgv][2d][q21][dtn]")
{ LaghosTest("imp_run_tgv_2d_q21_bwe_av0_dtn"); }

TEST_CASE("imp_run_sdv_2d_q21_mid_av0_dtn",
          "[dfem][imp][run][sdv][2d][q21][dtn]")
{ LaghosTest("imp_run_sdv_2d_q21_mid_av0_dtn"); }

TEST_CASE("imp_run_sdv_2d_q21_mid_av7_dtn",
          "[dfem][imp][run][sdv][2d][q21][dtn]")
{ LaghosTest("imp_run_sdv_2d_q21_mid_av7_dtn"); }
TEST_CASE("imp_run_sdv_2d_q21_mid_av1_dtn",
          "[dfem][imp][run][sdv][2d][q21][dtn]")
{ LaghosTest("imp_run_sdv_2d_q21_mid_av1_dtn"); }

TEST_CASE("imp_run_tgv_3d_q21_mid_av0_dtn",
          "[dfem][imp][run][tgv][3d][q21][dtn]")
{ LaghosTest("imp_run_tgv_3d_q21_mid_av0_dtn"); }

TEST_CASE("imp_run_sdv_3d_q21_mid_av7_dtn",
          "[dfem][imp][run][sdv][3d][q21][dtn]")
{ LaghosTest("imp_run_sdv_3d_q21_mid_av7_dtn"); }

// Problems 0 and 1 build a Cartesian mesh and must not open -m first.
TEST_CASE("generated_mesh_skips_file", "[dfem][mesh]")
{
   std::vector<std::string> args = {
      "laghos", "-p", "1", "-dim", "2", "-rs", "1", "-tf", "1e-4",
      "-m", "no-such-rectangle.mesh",
      "-no-glvis", "-no-paraview", "-no-save", "-vs", "1000"
   };
   std::vector<char *> argv;
   for (auto &a : args) { argv.push_back(&a[0]); }
   argv.push_back(nullptr);
   REQUIRE(laghos(static_cast<int>(argv.size()) - 1, argv.data()) == 0);
}

// A positive -oq must change the quadrature used by a Sedov step.
TEST_CASE("order_q_changes_energy", "[dfem][quad]")
{
   const std::string mesh = std::string(LAGHOS_DATA_DIR) + "square01_quad.mesh";
   const auto energy = [&](bool coarse)
   {
      std::vector<std::string> args = {
         "laghos", "-p", "1", "-dim", "2", "-rs", "1", "-s", "3", "-tf", "0.02",
         "-av", "-m", mesh,
         "-no-glvis", "-no-paraview", "-no-save", "-vs", "1000"
      };
      if (coarse)
      {
         args.push_back("-oq");
         args.push_back("1");
      }
      std::vector<char *> argv;
      for (auto &a : args) { argv.push_back(&a[0]); }
      argv.push_back(nullptr);
      LaghosResults res;
      laghos_results = &res;
      const int ret = laghos(static_cast<int>(argv.size()) - 1, argv.data());
      laghos_results = nullptr;
      REQUIRE(ret == 0);
      return res.energy_final;
   };
   REQUIRE(!AlmostEq(energy(true), energy(false), 1e-6));
}

// -av alone must change a Sedov step. Type 2 adds no viscous stress.
TEST_CASE("av_default_is_viscous", "[dfem][av]")
{
   const std::string mesh = std::string(LAGHOS_DATA_DIR) + "square01_quad.mesh";
   const auto energy = [&](const char *av)
   {
      std::vector<std::string> args = {
         "laghos", "-p", "1", "-dim", "2", "-rs", "1", "-s", "3", "-tf", "0.02",
         "-m", mesh, av, "-no-glvis", "-no-paraview", "-no-save", "-vs", "1000"
      };
      std::vector<char *> argv;
      for (auto &a : args) { argv.push_back(&a[0]); }
      argv.push_back(nullptr);
      LaghosResults res;
      laghos_results = &res;
      const int ret = laghos(static_cast<int>(argv.size()) - 1, argv.data());
      laghos_results = nullptr;
      REQUIRE(ret == 0);
      return res.energy_final;
   };
   REQUIRE(!AlmostEq(energy("-av"), energy("-no-av"), 1e-6));
}

// Taylor-Green must reject a Newton step that does not converge.
// The debug device faults while destroying a Newton solve that returned
// early, the same class of host read as the existing debug skip.
TEST_CASE("imp_tgv_newton_fail", "[dfem][imp][newton]")
{
   if (device == "debug") { return; }
   const std::string mesh = std::string(LAGHOS_DATA_DIR) + "square01_quad.mesh";
   std::vector<std::string> args = {
      "laghos", "-p", "0", "-dim", "2", "-rs", "1", "-s", "12", "-tf", "0.01",
      "-nmi", "1", "-nrt", "1e-30", "-nretry", "0", "-m", mesh,
      "-no-glvis", "-no-paraview", "-no-save", "-vs", "1000"
   };
   std::vector<char *> argv;
   for (auto &a : args) { argv.push_back(&a[0]); }
   argv.push_back(nullptr);
   REQUIRE(laghos(static_cast<int>(argv.size()) - 1, argv.data()) != 0);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) try
{
   mfem::Mpi::Init(argc, argv);

   std::vector<char*> args;
   for (int i = 0; i < argc; i++)
   {
      if (!std::strcmp(argv[i], "-v") || !std::strcmp(argv[i], "--verbose"))
      {
         verbose = true;
      }
      else if (!std::strcmp(argv[i], "--device") && i + 1 < argc)
      {
         device = argv[++i];
      }
      else if (!std::strcmp(argv[i], "--gpus")) { gpus = true; }
      else if (!std::strcmp(argv[i], "--qf") && i + 1 < argc)
      {
         qf = argv[++i];
      }
      else { args.push_back(argv[i]); }
   }

   const auto gpu = [](const char *backend)
   { return device.find(backend) != std::string::npos; };
   const bool gpu_device = gpu("cuda") || gpu("hip");

   // Only the GPU count with --gpus, 0 without a GPU device
   if (gpus)
   {
      if (mfem::Mpi::Root())
      {
         std::cout << (gpu_device ? mfem::Device::GetDeviceCount() : 0)
                   << std::endl;
      }
      return 0;
   }

   // One GPU per rank of a node, round-robin over the visible ones from the
   // first one, LAGHOS_TESTS_GPU or 0
   if (gpu_device)
   {
      MPI_Comm node;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &node);
      int local_rank;
      MPI_Comm_rank(node, &local_rank);
      MPI_Comm_free(&node);
      const char *first = std::getenv("LAGHOS_TESTS_GPU");
      device_id = ((first ? std::atoi(first) : 0) + local_rank) %
                  mfem::Device::GetDeviceCount();
   }

   // One Device per process, kept by all the laghos runs of the test cases
   mfem::Device backend(device, device_id);

   // Catch2 colors follow report::UseColor (TTY and no NO_COLOR), unless set
   static char colour_flag[] = "--use-colour", yes[] = "yes", no[] = "no";
   const bool colour_set = std::any_of(args.begin(), args.end(), [](char *arg)
   { return !std::strcmp(arg, "--use-colour"); });
   if (!colour_set)
   {
      args.push_back(colour_flag);
      args.push_back(report::UseColor() ? yes : no);
   }

   // Only rank 0 prints the Catch2 report
   static char out_flag[] = "--out", dev_null[] = "/dev/null";
   if (!mfem::Mpi::Root())
   {
      args.push_back(out_flag);
      args.push_back(dev_null);
   }
   const int ret = Catch::Session().run(static_cast<int>(args.size()),
                                        args.data());
   if (mfem::Mpi::Root() && summary.total > 0)
   {
      report::Summary("laghos_tests", summary.passed, summary.total);
   }
   return ret;
}
catch (std::exception& e)
{
   std::cerr << "\x1b[31m" << e.what() << "\x1b[m" << std::endl;
   return EXIT_FAILURE;
}
