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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <unistd.h>

#include "general/communication.hpp"
using mfem::real_t;
using args_ptr_t = std::vector<std::unique_ptr<char[]>>;
using args_t = std::vector<char*>;

int remhos(int, char *[], double &);

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
   static constexpr const char *common = "-no-vis";
   const std::string name, mesh, options;
   const size_t step;
   const real_t dt, energy = std::numeric_limits<real_t>::signaling_NaN();
   Test(const char *name, const char *mesh, const char *options,
        const size_t step, const real_t dt, const real_t energy):
      name(name), mesh(mesh), options(std::string(options) + " " + common),
      step(step), dt(dt), energy(energy) {}
   std::string Command() const
   {
      return std::string(binary) + " " + mesh + " " + options;
   }
};

///////////////////////////////////////////////////////////////////////////////
const Test runs[] =
{
   {
      "#1",
      "-m ../data/square01_quad.mesh", "-p 0 -rs 3 -tf 0.75 -pa",
      339, 0.000702, 0.0971135253228191
   }
};
constexpr int N_TESTS = sizeof(runs) / sizeof(Test);

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

   real_t energy{};
   remhos(args.size()-1, args.data(), energy);

   if (AlmostEq(energy, test.energy)) { return EXIT_SUCCESS; }

   mfem::err << "❌ " << test.name
             << std::fixed << std::setprecision(16)
             << ": energy: " << energy << " vs. " << test.energy
             << std::endl;
   return EXIT_FAILURE;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) try
{
   mfem::Mpi::Init(argc, argv);

   int opt;
   int test = -1;
   auto show_usage = [](const int ret = EXIT_FAILURE)
   {
      printf("Usage: program [-a <arg>] [-b <arg>] [-h]\n");
      printf("  -t <test>  Optional test number \n");
      printf("  -h         Show this help message\n");
      exit(ret);
   };

   while ((opt = getopt(argc, argv, "t:h")) != -1)
   {
      switch (opt)
      {
         case 't': test = std::atoi(optarg); break;
         case 'h': show_usage(EXIT_SUCCESS);
         default: show_usage(EXIT_FAILURE);
      }
   }

   if (test >= 0 && test < N_TESTS) { return LaghosTest(runs[test]); }

   for (auto & run : runs)
   {
      if (LaghosTest(run) != EXIT_SUCCESS) { return EXIT_FAILURE; }
   }
   return EXIT_SUCCESS;
}
catch (std::exception& e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
