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
// Non-regression tests of the dFEM Laghos:
//   laghos_tests            runs all the tests
//   laghos_tests -t <name>  runs one test, as done by CTest

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "laghos_tests.hpp"

using mfem::real_t;
using args_ptr_t = std::vector<std::unique_ptr<char[]>>;
using args_t = std::vector<char*>;

#ifndef LAGHOS_DATA_DIR
#define LAGHOS_DATA_DIR "../data/"
#endif

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
   static constexpr const char *common = "-no-glvis -no-paraview -vs 1000";
   static constexpr real_t tolerance = 1e-10;
   const std::string name, mesh, options;
   const int steps;
   const real_t dt, energy, x_norm, v_norm, e_norm;
   Test(const char *name, const char *mesh, const char *options,
        const int steps, const real_t dt, const real_t energy,
        const real_t x_norm, const real_t v_norm, const real_t e_norm):
      name(name), mesh(mesh), options(std::string(options) + " " + common),
      steps(steps), dt(dt), energy(energy),
      x_norm(x_norm), v_norm(v_norm), e_norm(e_norm) {}
   std::string Command() const
   {
      return std::string(binary) + " -m " + LAGHOS_DATA_DIR + mesh + " " +
             options;
   }
};

///////////////////////////////////////////////////////////////////////////////
// Explicit path only (-s <= 10, the RK solvers calling hydro->Mult).
// p0/p1 build their own Cartesian mesh, the -m mesh is only read.
// Reference values: steps, dt, final energy, |x|, |v|, |e| (true dofs l2 norms)
// MFEM: 01-dfem-laghos-dev-av (dfem-laghos-dev-av-macos), 7cc9281ce8
// constexpr real_t NaN = std::numeric_limits<real_t>::quiet_NaN();
const Test runs[] =
{
   {
      "sedov_2d", "square01_quad.mesh",
      "-p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.05",
      23,
      1.5436933912796560e-04, 2.5004931952471232e-01,
      1.4101829631350940e+01, 4.4195254279849605e+00, 4.2208594277324963e+01
   },
   {
      "sedov_2d_rk2", "square01_quad.mesh",
      "-p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 2 -cfl 0.5 -av -av-type 7 -tf 0.05",
      6,
      6.2443204309436178e-03, 2.4980489638517073e-01,
      7.5779033060146155e+00, 1.5104070411987085e+00, 1.4249329026179177e+01
   },
   {
      "sedov_3d", "cube01_hex.mesh",
      "-p 1 -dim 3 -rs 1 -ov 2 -oe 1 -s 3 -cfl 0.5 -av -av-type 7 -tf 0.02",
      6,
      9.7893419572273734e-06, 1.2501143644784798e-01,
      2.7832196671533371e+01, 3.3771156618999734e+00, 5.7714521901242946e+01
   },
   {
      "taylor_green_2d", "square01_quad.mesh",
      "-p 0 -dim 2 -rs 2 -ov 3 -oe 2 -s 4 -cfl 0.5 -tf 0.05",
      3,
      5.3791888466399942e-03, 1.7499984739484764e+00,
      1.0847749476402059e+01, 9.1602573868358732e+00, 1.8694024191662098e+01
   },
   {
      "taylor_green_3d", "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05",
      7,
      1.9746163022951384e-03, 1.4993325901772420e+02,
      1.1859991863857640e+01, 6.2046602782719855e+00, 1.1985122651149325e+03
   },
   {
      "taylor_green_3d_av", "cube01_hex.mesh",
      "-p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -av -av-type 7 -tf 0.05",
      18,
      1.4960934056207102e-03, 1.4993325988985328e+02,
      1.1858399469894451e+01, 9.9669267367608894e-01, 1.1994458165617593e+03
   },
};

///////////////////////////////////////////////////////////////////////////////
int LaghosTest(const Test & test)
{
   static args_ptr_t args_ptr;
   args_t args;

   std::istringstream iss(test.Command());

   std::string token;
   while (iss >> token)
   {
      auto arg_ptr = std::make_unique<char[]>(token.size() + 1);
      std::memcpy(arg_ptr.get(), token.c_str(), token.size() + 1);
      arg_ptr[token.size()] = '\0';
      args.push_back(arg_ptr.get());
      args_ptr.emplace_back(std::move(arg_ptr));
   }
   args.push_back(nullptr);

   LaghosResults res;
   if (laghos(args.size()-1, args.data(), res) != 0) { return EXIT_FAILURE; }

   bool success = res.steps == test.steps;
   std::ostringstream os;
   os << std::scientific << std::setprecision(16);
   os << "   steps:  " << res.steps << " vs. " << test.steps
      << (res.steps == test.steps ? "" : "  <--") << "\n";
   auto check = [&](const char *name, const real_t x, const real_t y)
   {
      const bool ok = AlmostEq(x, y, Test::tolerance);
      os << "   " << name << x << " vs. " << y << (ok ? "" : "  <--") << "\n";
      success &= ok;
   };
   check("dt:     ", res.dt, test.dt);
   check("energy: ", res.energy_final, test.energy);
   check("|x|:    ", res.x_norm, test.x_norm);
   check("|v|:    ", res.v_norm, test.v_norm);
   check("|e|:    ", res.e_norm, test.e_norm);
   // Same layout as the runs[] reference values, to update them if needed
   os << "   ref: " << res.steps << ", " << res.dt << ", " << res.energy_final
      << ", " << res.x_norm << ", " << res.v_norm << ", " << res.e_norm << "\n";

   if (mfem::Mpi::Root())
   {
      mfem::out << (success ? "✅ " : "❌ ") << test.name << ": "
                << test.Command() << "\n" << os.str() << std::flush;
   }
   return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) try
{
   mfem::Mpi::Init(argc, argv);

   int opt;
   const char *test = nullptr;
   auto show_usage = [](const int ret = EXIT_FAILURE)
   {
      printf("Usage: laghos_tests [-t <name>] [-h]\n");
      printf("  -t <name>  Optional test name, all tests otherwise\n");
      printf("  -h         Show this help message\n");
      exit(ret);
   };

   while ((opt = getopt(argc, argv, "t:h")) != -1)
   {
      switch (opt)
      {
         case 't': test = optarg; break;
         case 'h': show_usage(EXIT_SUCCESS);
         default: show_usage(EXIT_FAILURE);
      }
   }

   if (test)
   {
      for (auto & run : runs)
      {
         if (run.name == test) { return LaghosTest(run); }
      }
      mfem::err << "Unknown test: " << test << std::endl;
      return EXIT_FAILURE;
   }

   int failures = 0;
   for (auto & run : runs)
   {
      if (LaghosTest(run) != EXIT_SUCCESS) { failures++; }
   }
   return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch (std::exception& e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
