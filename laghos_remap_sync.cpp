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

#include "laghos_remap_sync.hpp"

#define EMPTY_ZONE_TOL 1e-12

using namespace std;
namespace mfem
{
namespace ale
{


void ComputeBoolIndicators(int NE, const Vector &u,
                           Array<bool> &ind_elem, Array<bool> &ind_dofs)
{
   ind_elem.SetSize(NE);
   ind_dofs.SetSize(u.Size());

   ind_elem.HostWrite();
   ind_dofs.HostWrite();
   u.HostRead();

   const int ndof = u.Size() / NE;
   int dof_id;
   for (int i = 0; i < NE; i++)
   {
      ind_elem[i] = false;
      for (int j = 0; j < ndof; j++)
      {
         dof_id = i*ndof + j;
         ind_dofs[dof_id] = (u(dof_id) > EMPTY_ZONE_TOL) ? true : false;

         if (u(dof_id) > EMPTY_ZONE_TOL) { ind_elem[i] = true; }
      }
   }
}

void ComputeRatio(int NE, const Vector &us, const Vector &u,
                  Vector &s, Array<bool> &bool_el, Array<bool> &bool_dof)
{
   ComputeBoolIndicators(NE, u, bool_el, bool_dof);

   us.HostRead();
   u.HostRead();
   s.HostWrite();
   bool_el.HostRead();
   bool_dof.HostRead();

   const int ndof = u.Size() / NE;
   for (int i = 0; i < NE; i++)
   {
      if (bool_el[i] == false)
      {
         for (int j = 0; j < ndof; j++) { s(i*ndof + j) = 0.0; }
         continue;
      }

      const double *u_el = &u(i*ndof), *us_el = &us(i*ndof);
      double *s_el = &s(i*ndof);

      // Average of the existing ratios. This does not target any kind of
      // conservation. The only goal is to have s_avg between the max and min
      // of us/u, over the active dofs.
      int n = 0;
      double sum = 0.0;
      for (int j = 0; j < ndof; j++)
      {
         if (bool_dof[i*ndof + j])
         {
            sum += us_el[j] / u_el[j];
            n++;
         }
      }
      MFEM_VERIFY(n > 0, "Major error that makes no sense");
      const double s_avg = sum / n;

      for (int j = 0; j < ndof; j++)
      {
         s_el[j] = (bool_dof[i*ndof + j]) ? us_el[j] / u_el[j] : s_avg;
      }
   }
}

void ZeroOutEmptyDofs(const Array<bool> &ind_elem,
                      const Array<bool> &ind_dofs, Vector &u)
{
   ind_elem.HostRead();
   ind_dofs.HostRead();
   u.HostReadWrite();

   const int NE = ind_elem.Size();
   const int ndofs = u.Size() / NE;
   for (int k = 0; k < NE; k++)
   {
      if (ind_elem[k] == true) { continue; }

      for (int i = 0; i < ndofs; i++)
      {
         if (ind_dofs[k*ndofs + i] == false) { u(k*ndofs + i) = 0.0; }
      }
   }
}

} // namespace ale
} // namespace mfem
