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

namespace mfem
{

namespace hydrodynamics
{
   // **************************************************************************
   static void offsetNindices(ParFiniteElementSpace &fes,
                              mfem::Array<int> &offsets,
                              mfem::Array<int> &indices){
      const int elements = fes.GetNE();
      const int globalDofs = fes.GetNDofs();
      const int localDofs = fes.GetFE(0)->GetDof();
      const FiniteElement *fe = fes.GetFE(0);
      const TensorBasisElement* el = dynamic_cast<const TensorBasisElement*>(fe);
      const Array<int> &dof_map = el->GetDofMap();
      const bool dof_map_is_identity = dof_map.Size()==0;
      const Table& e2dTable = fes.GetElementToDofTable();
      const int *elementMap = e2dTable.GetJ();      
      Array<int> h_offsets(globalDofs+1);
      // We'll be keeping a count of how many local nodes point to its global dof
      for (int i = 0; i <= globalDofs; ++i) {
         h_offsets[i] = 0;
      }
      for (int e = 0; e < elements; ++e) {
         for (int d = 0; d < localDofs; ++d) {
            const int gid = elementMap[localDofs*e + d];
            ++h_offsets[gid + 1];
         }
      }
      // Aggregate to find offsets for each global dof
      for (int i = 1; i <= globalDofs; ++i) {
         h_offsets[i] += h_offsets[i - 1];
      }
      Array<int> h_indices(localDofs*elements);
      // For each global dof, fill in all local nodes that point   to it
      for (int e = 0; e < elements; ++e) {
         for (int d = 0; d < localDofs; ++d) {
            const int did = dof_map_is_identity?d:dof_map[d];
            const int gid = elementMap[localDofs*e + did];
            const int lid = localDofs*e + d;
            h_indices[h_offsets[gid]++] = lid;
         }
      }
      // We shifted the offsets vector by 1 by using it as a counter
      // Now we shift it back.
      for (int i = globalDofs; i > 0; --i) {
         h_offsets[i] = h_offsets[i - 1];
      }
      h_offsets[0] = 0;  
      offsets = h_offsets;
      indices = h_indices;
   }
   
   // **************************************************************************
   static void globalToLocal0(const int vdim,
                              const bool ordering,
                              const int globalEntries,                                           
                              const int localEntries,
                              const int* __restrict offsets,
                              const int* __restrict indices,
                              const double* __restrict globalX,
                              double* __restrict localX) {
      for(int i=1; i<globalEntries; i+=1) {
         const int offset = offsets[i];
         const int nextOffset = offsets[i+1];
         for (int v = 0; v < vdim; ++v) {
            const int g_offset = ijNMt(v,i,vdim,globalEntries,ordering);
            const double dofValue = globalX[g_offset];
            for (int j = offset; j < nextOffset; ++j) {
               const int l_offset = ijNMt(v,indices[j],vdim,localEntries,ordering);
               localX[l_offset] = dofValue;
            }
         }
      }
   }
   
   // **************************************************************************
   void globalToLocal(ParFiniteElementSpace &fes,
                      const double *globalVec,
                      double *localVec,
                      const bool ordering) {
      const int vdim = fes.GetVDim(); assert(vdim==2);
      const int localDofs = fes.GetFE(0)->GetDof();
      const int globalDofs = fes.GetNDofs();
      const int localEntries = localDofs * fes.GetNE();
      mfem::Array<int> offsets,indices;
      offsetNindices(fes,offsets,indices); // should be stored
      //dbg("offsets:");offsets.Print();
      //dbg("indices:");indices.Print();
      globalToLocal0(vdim, ordering,
                     globalDofs, localEntries,
                     offsets, indices,
                     globalVec, localVec);
   }   
   
} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
