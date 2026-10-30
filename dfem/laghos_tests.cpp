// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
// Non-regression tests of the dFEM Laghos explicit path, with the Catch2 of
// MFEM's tests/unit/catch.hpp:
//
//   laghos_tests                  runs all the tests
//   laghos_tests sedov_2d         runs one test by name, as done by CTest
//   laghos_tests "[taylor-green]" runs the tests with a given tag
//   laghos_tests -v ...           also shows the raw Laghos output
//   laghos_tests -h               Catch2 help

#include <algorithm>
#include <cmath>
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
   static constexpr real_t tolerance = 1e-10;
   const std::string name, desc, mesh, options;
   const int steps;
   const real_t dt, energy, x_norm, v_norm, e_norm;
   Test(const char *name, const char *desc, const char *mesh,
        const char *options,
        const int steps, const real_t dt, const real_t energy,
        const real_t x_norm, const real_t v_norm, const real_t e_norm):
      name(name), desc(desc), mesh(mesh),
      options(std::string(options) + " " + common),
      steps(steps), dt(dt), energy(energy),
      x_norm(x_norm), v_norm(v_norm), e_norm(e_norm) {}
   std::string Command() const
   {
      return std::string(binary) + " -m " + LAGHOS_DATA_DIR + mesh + " " +
             options;
   }
};

///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      "sedov_2d", "explicit RK3SSP, Q2-Q1, av-type 7, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.05",
      23,
      1.5436933912796560e-04, 2.5004931952471232e-01,
      1.4101829631350940e+01, 4.4195254279849605e+00, 4.2208594277324963e+01
   },
   {
      "sedov_2d_rk2", "explicit RK2, Q2-Q1, av-type 7, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 2 -cfl 0.5 -av -av-type 7 -tf 0.05",
      6,
      6.2443204309436178e-03, 2.4980489638517073e-01,
      7.5779033060146155e+00, 1.5104070411987085e+00, 1.4249329026179177e+01
   },
   {
      "sedov_3d", "explicit RK3SSP, Q2-Q1, av-type 7, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.02",
      6,
      9.7893419572273734e-06, 1.2501143644784798e-01,
      2.7832196671533371e+01, 3.3771156618999734e+00, 5.7714521901242946e+01
   },
   // Stock-comparable Sedov: stock Laghos tensor viscosity (-av-type 1) and
   // time step estimate (-stock-dt), same steps, dt, ΔE and |e| as
   //   laghos -dim 2 -nx 1 -ny 1 -rs 3 -p 1 -ok 2 -ot 1 -s 3 -cfl 0.5 -tf 0.05 -fa
   //   laghos -dim 3 -nx 2 -ny 2 -nz 2 -rs 1 -p 1 -ok 2 -ot 1 -s 3 -cfl 0.5 -tf 0.02 -fa
   {
      "sedov_2d_stock", "explicit RK3SSP, Q2-Q1, stock av, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.05",
      22, 1.5244822594611736e-03, 2.5005546874349194e-01, 1.4102287121585931e+01,
      4.5117390396546329e+00, 4.0702975696596390e+01
   },
   {
      "sedov_3d_stock", "explicit RK3SSP, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.02",
      4, 4.3725299546347671e-03, 1.2503911284057320e-01, 2.7832229285175657e+01,
      3.4262677669332566e+00, 5.6552559708003621e+01
   },
   {
      "taylor_green_2d", "explicit RK4, Q3-Q2, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 3 -oe 2 -s 4 -cfl 0.5 -tf 0.05",
      3,
      5.3791888466399942e-03, 1.7499984739484764e+00,
      1.0847749476402059e+01, 9.1602573868358732e+00, 1.8694024191662098e+01
   },
   {
      "taylor_green_3d", "explicit RK4, Q2-Q1, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05",
      7,
      1.9746163022951384e-03, 1.4993325901494498e+02,
      1.1859990900590070e+01, 6.2046596711615782e+00, 1.1985122282081534e+03
   },
   {
      "taylor_green_3d_av", "explicit RK4, Q2-Q1, av-type 7, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 7 -tf 0.05",
      18,
      1.4960934056207102e-03, 1.4993325988669309e+02,
      1.1858399916887008e+01, 9.9668537477810759e-01, 1.1994457713114568e+03
   },
   // Smaller runs of the README 'Verification of Results' cases, with stock
   // defaults (Q2-Q1, RK4, cfl 0.5), compared with (data/ meshes, -pa)
   //   laghos -p 0 -m square01_quad.mesh -rs 1 -tf 0.05   (README 1)
   //   laghos -p 0 -m cube01_hex.mesh -rs 1 -tf 0.01      (README 2)
   //   laghos -p 1 -m square01_quad.mesh -rs 2 -tf 0.05   (README 3)
   //   laghos -p 1 -m cube01_hex.mesh -rs 1 -tf 0.02      (README 4, -E0 1)
   // dFEM Laghos starts from a 1x1 square, one more -rs than the 2x2 mesh
   {
      "readme_taylor_green_2d", "README 1, explicit RK4, Q2-Q1, no av, 2D",
      "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05",
      2, 1.7273414632041139e-02, 1.7496138442784803e+00, 7.5755635427647361e+00,
      6.3210512132666858e+00, 1.2508147843915047e+01
   },
   {
      "readme_taylor_green_3d", "README 2, explicit RK4, Q2-Q1, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 1 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.01",
      3, 1.8178734530672098e-03, 1.4993721050652641e+02, 2.7831009958537454e+01,
      1.4156587512698188e+01, 3.3898742905766835e+03
   },
   {
      "readme_sedov_2d", "README 3, explicit RK4, Q2-Q1, stock av, 2D",
      "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.05",
      21, 1.8984422273346600e-03, 2.5000401709103320e-01, 1.4102285371332647e+01,
      4.5108287429679663e+00, 4.0693374548498461e+01
   },
   {
      "readme_sedov_3d", "README 4, explicit RK4, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.02",
      4, 4.3725299546347671e-03, 1.2500220220066449e-01, 2.7832227632579837e+01,
      3.4200678868617302e+00, 5.6528006524554613e+01
   },
   // More 3D: lowest order Q1-Q0, stock av on a smooth flow, and the triple
   // point, the only problem reading its -m mesh (box with two materials),
   // stock-comparable runs as
   //   laghos -p 0 -dim 3 -m cube01_hex.mesh -rs 1 -ok 1 -ot 0 -s 4 -tf 0.05
   //   laghos -p 1 -dim 3 -m cube01_hex.mesh -rs 1 -ok 1 -ot 0 -s 3 -tf 0.1
   //   laghos -p 0 -dim 3 -m cube01_hex.mesh -rs 0 -ok 2 -ot 1 -s 4 -iv -tf 0.05
   //   laghos -p 3 -dim 3 -m box01_hex.mesh -rs 0 -ok 2 -ot 1 -s 4 -tf 0.2
   {
      "taylor_green_3d_q1q0", "explicit RK4, Q1-Q0, no av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 1 -ov 1 -oe 0 -s 4 -cfl 0.5 -tf 0.05",
      7, 6.6060646756573873e-04, 1.4990434628565481e+02, 1.1859894505936010e+01,
      5.5879520246439682e+00, 1.1985981660785569e+03
   },
   {
      "sedov_3d_q1q0_stock", "explicit RK3SSP, Q1-Q0, stock av, 3D",
      "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 1 -oe 0 -s 3 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.1",
      6, 4.5106785655475044e-03, 1.2526730824116419e-01, 1.1868845859494494e+01,
      1.9848019446683176e+00, 6.8841771094494320e+00
   },
   {
      "taylor_green_3d_stock_av", "explicit RK4, Q2-Q1, stock av, 3D",
      "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.05",
      14, 1.1438157810123699e-03, 1.4993326044731037e+02, 1.1858537217399940e+01,
      1.7347628571356015e+00, 1.1993939640687433e+03
   },
   {
      "triple_point_3d", "explicit RK4, Q2-Q1, stock av, 3D",
      "box01_hex.mesh",
      "-p 3 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 1 -stock-dt -tf 0.2",
      5, 3.2015046545843445e-02, 3.0150228406430880e+01, 7.0909362959399473e+01,
      2.1011239969290898e+00, 1.6921507332503591e+01
   },
   {
      "triple_point_3d_av", "explicit RK4, Q2-Q1, av-type 7, 3D",
      "box01_hex.mesh",
      "-p 3 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 7 -tf 0.2",
      10, 1.0248581728371387e-02, 3.0150198118370763e+01, 7.0902482097110010e+01,
      1.3479830805551927e+00, 1.6939266983223721e+01
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

   // Quiet Laghos run, unless -v
   LaghosResults res;
   const bool out_enabled = mfem::out.IsEnabled();
   if (!verbose) { mfem::out.Disable(); }
   laghos_results = &res;
   const int ret = laghos(static_cast<int>(args.size()) - 1, args.data());
   laghos_results = nullptr;
   if (out_enabled) { mfem::out.Enable(); }

   const bool root = mfem::Mpi::Root();
   if (root) { report::Header(test.name, test.desc); }
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
      {"dt", res.dt, test.dt, AlmostEq(res.dt, test.dt, Test::tolerance)},
      {"E", res.energy_final, test.energy, AlmostEq(res.energy_final, test.energy, Test::tolerance)},
      {"‖x‖", res.x_norm, test.x_norm, AlmostEq(res.x_norm, test.x_norm, Test::tolerance)},
      {"‖v‖", res.v_norm, test.v_norm, AlmostEq(res.v_norm, test.v_norm, Test::tolerance)},
      {"‖e‖", res.e_norm, test.e_norm, AlmostEq(res.e_norm, test.e_norm, Test::tolerance)},
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
         row.Rel("rel", rel_max, Test::tolerance).Mark(ok);
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
            .Num("ref", c.ref).Rel("rel", c.Rel(), Test::tolerance);
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
TEST_CASE("sedov_2d", "[dfem][explicit][sedov][2d]")
{ LaghosTest("sedov_2d"); }

TEST_CASE("sedov_2d_rk2", "[dfem][explicit][sedov][2d]")
{ LaghosTest("sedov_2d_rk2"); }

TEST_CASE("sedov_3d", "[dfem][explicit][sedov][3d]")
{ LaghosTest("sedov_3d"); }

TEST_CASE("sedov_2d_stock", "[dfem][explicit][sedov][2d][stock]")
{ LaghosTest("sedov_2d_stock"); }

TEST_CASE("sedov_3d_stock", "[dfem][explicit][sedov][3d][stock]")
{ LaghosTest("sedov_3d_stock"); }

TEST_CASE("taylor_green_2d", "[dfem][explicit][taylor-green][2d]")
{ LaghosTest("taylor_green_2d"); }

TEST_CASE("taylor_green_3d", "[dfem][explicit][taylor-green][3d]")
{ LaghosTest("taylor_green_3d"); }

TEST_CASE("taylor_green_3d_av", "[dfem][explicit][taylor-green][3d]")
{ LaghosTest("taylor_green_3d_av"); }

TEST_CASE("readme_taylor_green_2d",
          "[dfem][explicit][readme][taylor-green][2d]")
{ LaghosTest("readme_taylor_green_2d"); }

TEST_CASE("readme_taylor_green_3d",
          "[dfem][explicit][readme][taylor-green][3d]")
{ LaghosTest("readme_taylor_green_3d"); }

TEST_CASE("readme_sedov_2d", "[dfem][explicit][readme][sedov][2d]")
{ LaghosTest("readme_sedov_2d"); }

TEST_CASE("readme_sedov_3d", "[dfem][explicit][readme][sedov][3d]")
{ LaghosTest("readme_sedov_3d"); }

TEST_CASE("taylor_green_3d_q1q0", "[dfem][explicit][taylor-green][3d]")
{ LaghosTest("taylor_green_3d_q1q0"); }

TEST_CASE("sedov_3d_q1q0_stock", "[dfem][explicit][sedov][3d][stock]")
{ LaghosTest("sedov_3d_q1q0_stock"); }

TEST_CASE("taylor_green_3d_stock_av",
          "[dfem][explicit][taylor-green][3d][stock]")
{ LaghosTest("taylor_green_3d_stock_av"); }

TEST_CASE("triple_point_3d", "[dfem][explicit][triple-point][3d][stock]")
{ LaghosTest("triple_point_3d"); }

TEST_CASE("triple_point_3d_av", "[dfem][explicit][triple-point][3d]")
{ LaghosTest("triple_point_3d_av"); }

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
      else { args.push_back(argv[i]); }
   }

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
