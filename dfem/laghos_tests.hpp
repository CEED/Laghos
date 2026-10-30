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

#ifndef LAGHOS_DFEM_TESTS_HPP
#define LAGHOS_DFEM_TESTS_HPP

#include <mfem.hpp>

/// Final quantities of a dFEM Laghos run, checked by laghos_tests
struct LaghosResults
{
   int steps = 0;
   mfem::real_t t = 0.0, dt = 0.0;
   mfem::real_t energy_init = 0.0, energy_final = 0.0;
   mfem::real_t x_norm = 0.0, v_norm = 0.0, e_norm = 0.0;
};

/// laghos.cpp main, when compiled with LAGHOS_TESTS
int laghos(int argc, char *argv[], LaghosResults &results);

#endif // LAGHOS_DFEM_TESTS_HPP
