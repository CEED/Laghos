// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
extern "C" void vector_get_subvector(const int entries,
                                     double* __restrict v0,
                                     const double* v1,
                                     const int* v2) {
  for (int i = 0; i < entries; ++i) {
    const int dof_i = v2[i];
    v0[i] = dof_i >= 0 ? v1[dof_i] : -v1[-dof_i-1];
  }
}

