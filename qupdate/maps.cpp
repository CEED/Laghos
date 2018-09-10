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

#include "../laghos_solver.hpp"
#include "qupdate.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem {
   
namespace hydrodynamics {
   
   // **************************************************************************
   int *global2LocalMap(ParFiniteElementSpace &fes){
      const int elements = fes.GetNE();
      const int globalDofs = fes.GetNDofs();
      const int localDofs = fes.GetFE(0)->GetDof();

      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &dof_map = el->GetDofMap();
      const bool dof_map_is_identity = dof_map.Size()==0;
      const Table& e2dTable = fes.GetElementToDofTable();
      const int *elementMap = e2dTable.GetJ();
      Array<int> *h_map = new mfem::Array<int>(localDofs*elements);
      
      for (int e = 0; e < elements; ++e) {
         for (int d = 0; d < localDofs; ++d) {
            const int did = dof_map_is_identity?d:dof_map[d];
            const int gid = elementMap[localDofs*e + did];
            const int lid = localDofs*e + d;
            (*h_map)[lid] = gid;
         }
      }
      return h_map->GetData();
   }

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
