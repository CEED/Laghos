// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_mesh.hpp"
#include <unordered_map>

namespace mfem
{

namespace hydrodynamics
{

Mesh *CartesianMesh(int dim, int mpi_cnt, int elem_per_mpi, bool print,
                    int &par_ref, int **partitioning)
{
   MFEM_VERIFY(dim > 1, "Not implemented for 1D meshes.");

   auto factor = [&](int N)
   {
      for (int i = static_cast<int>(sqrt(N)); i > 0; i--)
      { if (N % i == 0) { return i; } }
      return 1;
   };

   par_ref = 0;
   const int ref_factor = (dim == 2) ? 4 : 8;

   // Elements per task before performing parallel refinements.
   // This will be used to form the serial mesh.
   int el0 = elem_per_mpi;
   while (el0 % ref_factor == 0)
   {
      el0 /= ref_factor;
      par_ref++;
   }

   // In the serial mesh we have:
   // The number of MPI blocks is mpi_cnt = mp_x.mpy_y.mpy_z.
   // The size of each MPI block is el0 = el0_x.el0_y.el0_z.
   int mpi_x, mpi_y, mpi_z;
   int el0_x, el0_y, el0_z;
   if (dim == 2)
   {
      mpi_x = factor(mpi_cnt);
      mpi_y = mpi_cnt / mpi_x;

      // Switch order for better balance.
      el0_y = factor(el0);
      el0_x = el0 / el0_y;
   }
   else
   {
      mpi_x = factor(mpi_cnt);
      mpi_y = factor(mpi_cnt / mpi_x);
      mpi_z = mpi_cnt / mpi_x / mpi_y;

      // Switch order for better balance.
      el0_z = factor(el0);
      el0_y = factor(el0 / el0_z);
      el0_x = el0 / el0_y / el0_z;
   }

   if (print && dim == 2)
   {
      int elem_par_x = mpi_x * el0_x * pow(2, par_ref),
          elem_par_y = mpi_y * el0_y * pow(2, par_ref);

      std::cout << "--- Mesh generation: \n";
      std::cout << "Par mesh:    " << elem_par_x << " x " << elem_par_y
                << " (" << elem_par_x * elem_par_y << " elements)\n"
                << "Elem / task: "
                << el0_x * pow(2, par_ref) << " x "
                << el0_y * pow(2, par_ref)
                << " (" << el0_x * pow(2, 2*par_ref) * el0_y << " elements)\n"
                << "MPI blocks:  " << mpi_x << " x " << mpi_y
                << " (" << mpi_x * mpi_y << " mpi tasks)\n" << "-\n"
                << "Serial mesh: "
                << mpi_x * el0_x << " x " << mpi_y * el0_y
                << " (" << mpi_x * el0_x * mpi_y * el0_y << " elements)\n"
                << "Elem / task: " << el0_x << " x " << el0_y << std::endl
                << "Par refine:  " << par_ref << std::endl;
      std::cout << "--- \n";
   }

   if (print && dim == 3)
   {
      int elem_par_x = mpi_x * el0_x * pow(2, par_ref),
          elem_par_y = mpi_y * el0_y * pow(2, par_ref),
          elem_par_z = mpi_z * el0_z * pow(2, par_ref);

      std::cout << "--- Mesh generation: \n";
      std::cout << "Par mesh:    "
                << elem_par_x << " x " << elem_par_y << " x " << elem_par_z
                << " (" << elem_par_x*elem_par_y*elem_par_z << " elements)\n"
                << "Elem / task: "
                << el0_x * pow(2, par_ref) << " x "
                << el0_y * pow(2, par_ref) << " x "
                << el0_z * pow(2, par_ref)
                << " (" << el0_x*pow(2, 3*par_ref)*el0_y*el0_z << " elements)\n"
                << "MPI blocks:  " << mpi_x << " x " << mpi_y << " x " << mpi_z
                << " (" << mpi_x * mpi_y * mpi_z << " mpi tasks)\n" << "-\n"
                << "Serial mesh: "
                << mpi_x*el0_x << " x " << mpi_y*el0_y << " x " << mpi_z*el0_z
                << " (" << mpi_x*el0_x*mpi_y*el0_y*mpi_z*el0_z << " elements)\n"
                << "Elem / task: "
                << el0_x << " x " << el0_y << " x " << el0_z  << std::endl
                << "Par refine:  " << par_ref << std::endl;
      std::cout << "--- \n";
   }

   Mesh *mesh;
   if (dim == 2)
   {
      mesh = new Mesh(Mesh::MakeCartesian2D(mpi_x * el0_x,
                                            mpi_y * el0_y,
                                            Element::QUADRILATERAL, true));
      // Fix attributes.
      const int NBE = mesh->GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         const int attr_old = mesh->GetBdrElement(b)->GetAttribute();
         int attr_new = -1;
         if (attr_old == 1 || attr_old == 3) { attr_new = 2; }
         if (attr_old == 2 || attr_old == 4) { attr_new = 1; }
         MFEM_VERIFY(attr_new > 0, "Attribute error.");
         mesh->GetBdrElement(b)->SetAttribute(attr_new);
      }
   }
   else
   {
      mesh = new Mesh(Mesh::MakeCartesian3D(mpi_x * el0_x,
                                            mpi_y * el0_y,
                                            mpi_z * el0_z,
                                            Element::HEXAHEDRON, true));
      // Fix attributes.
      const int NBE = mesh->GetNBE();
      for (int b = 0; b < NBE; b++)
      {
         const int attr_old = mesh->GetBdrElement(b)->GetAttribute();
         int attr_new = -1;
         if (attr_old == 1 || attr_old == 6) { attr_new = 3; }
         if (attr_old == 3 || attr_old == 5) { attr_new = 1; }
         if (attr_old == 2 || attr_old == 4) { attr_new = 2; }
         MFEM_VERIFY(attr_new > 0, "Attribute error.");
         mesh->GetBdrElement(b)->SetAttribute(attr_new);
      }
   }

   auto nxyz = new int[dim];
   if (dim == 2) { nxyz[0] = mpi_x; nxyz[1] = mpi_y; }
   else          { nxyz[0] = mpi_x; nxyz[1] = mpi_y; nxyz[2] = mpi_z; }
   *partitioning = mesh->CartesianPartitioning(nxyz);

   delete[] nxyz;
   return mesh;
}


} // namespace hydrodynamics

} // namespace mfem
