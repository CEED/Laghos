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

#ifndef LAGHOS_DFEM_HPP
#define LAGHOS_DFEM_HPP

#include <vector>
#include <mfem.hpp>

/// Quantities of a dFEM Laghos run, filled when given to laghos()
struct LaghosResults
{
   struct Step { int ti; mfem::real_t t, dt; };
   long long ne = 0, h1_dofs = 0, l2_dofs = 0; // global sizes
   mfem::real_t dt_init = 0.0; // initial time step estimate
   int repeats = 0; // repeated time steps
   std::vector<Step> history; // accepted time steps
   int steps = 0;
   mfem::real_t t = 0.0, dt = 0.0;
   mfem::real_t energy_init = 0.0, energy_final = 0.0;
   mfem::real_t x_norm = 0.0, v_norm = 0.0, e_norm = 0.0;
};

/// dFEM Laghos run with the command line arguments, see laghos_main.cpp,
/// filling results when not null, as for laghos_tests.cpp
int laghos(int argc, char *argv[], LaghosResults *results = nullptr);

#endif // LAGHOS_DFEM_HPP
