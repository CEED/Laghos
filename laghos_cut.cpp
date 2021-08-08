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

#include "laghos_cut.hpp"

namespace mfem
{

namespace hydrodynamics
{

using namespace std;

// Used for debugging the elem-to-dof tables when the elements' attributes
// are associated with materials. Options for lvl:
// 0 - only duplicated materials per DOF.
// 1 - all materials per DOF .
// 2 - full element/material output per DOF.
void PrintDofElemTable(const Table &elem_dof, const ParMesh &pmesh,
                       int lvl, bool boundary)
{
   Table dof_elem;
   Transpose(elem_dof, dof_elem);

   const int nrows = dof_elem.Size();
   if (boundary == false)
   {
      std::cout << "--- Dof-to-Elem. Total elem DOFs: " << nrows << std::endl;
   }
   else
   {
      std::cout << "--- Dof-to-Bdr. Total bndry DOFs: " << nrows << std::endl;
   }

   Array<int> dof_elements;
   for (int dof = 0; dof < nrows; dof++)
   {
      // Find the materials that share the current dof.
      std::set<int> dof_materials;
      dof_elem.GetRow(dof, dof_elements);
      if (lvl == 2) { std::cout << "Elements for DOF " << dof << ": \n"; }
      for (int e = 0; e < dof_elements.Size(); e++)
      {
         int mat_id;
         if (boundary == false)
         {
            mat_id = pmesh.GetAttribute(dof_elements[e]);
         }
         else
         {
            int face_id = pmesh.GetBdrFace(dof_elements[e]);
            int elem_id, tmp;
            pmesh.GetFaceElements(face_id, &elem_id, &tmp);
            mat_id = pmesh.GetAttribute(elem_id);
         }

         if (lvl == 2) { cout << dof_elements[e] << "(" << mat_id << ") "; }

         dof_materials.insert(mat_id);
      }
      if (lvl == 2) { std::cout << std::endl; }

      if (lvl == 2) { continue; }
      if (lvl == 0 && dof_materials.size() < 2) { continue; }

      std::cout << "Materials for DOF " << dof << ": " << std::endl;
      for (auto it = dof_materials.cbegin(); it != dof_materials.cend(); it++)
      { std::cout << *it << ' '; }
      std::cout << std::endl;
   }
   std::cout << "--- End of Table" << std::endl;
}

void cutH1Space(ParFiniteElementSpace &pfes, bool vis, bool print)
{
   ParMesh &pmesh = *pfes.GetParMesh();
   ParGridFunction x_vis(&pfes);

   // Duplicate DOFs on the material interface.
   // That is, the DOF touches different element attributes.
   const Table &elem_dof = pfes.GetElementToDofTable(),
               &bdre_dof = pfes.GetBdrElementToDofTable();
   Table dof_elem, dof_bdre;
   Table new_elem_dof(elem_dof), new_bdre_dof(bdre_dof);
   Transpose(elem_dof, dof_elem);
   Transpose(bdre_dof, dof_bdre);
   const int nrows = dof_elem.Size(), n_bdr_dofs = dof_bdre.Size();
   int ndofs = nrows;
   Array<int> dof_elements, dof_boundaries;
   if (print)
   {
      PrintDofElemTable(elem_dof, pmesh, 2, false);
      PrintDofElemTable(bdre_dof, pmesh, 2, true);
   }
   for (int dof = 0; dof < nrows; dof++)
   {
      // Check which materials share the current dof.
      std::set<int> dof_materials;
      dof_elem.GetRow(dof, dof_elements);
      for (int e = 0; e < dof_elements.Size(); e++)
      {
         const int mat_id = pmesh.GetAttribute(dof_elements[e]);
         dof_materials.insert(mat_id);
      }
      // Count the materials for the current DOF.
      const int dof_mat_cnt = dof_materials.size();

      // Duplicate the dof if it is shared between materials.
      if (dof_mat_cnt > 1)
      {
         // The material with the lowest index keeps the old DOF id.
         // All other materials duplicate the dof.
         auto mat = dof_materials.cbegin();
         mat++;
         while(mat != dof_materials.cend())
         {
            // Replace in all elements with material mat.
            const int new_dof_id = ndofs;
            for (int e = 0; e < dof_elements.Size(); e++)
            {
               if (pmesh.GetAttribute(dof_elements[e]) == *mat)
               {
                  if (print)
                  {
                     std::cout << "Replacing DOF (for element) : "
                               << dof << " -> " << new_dof_id
                               << " in EL " << dof_elements[e] << std::endl;
                  }
                  new_elem_dof.ReplaceConnection(dof_elements[e],
                                                 dof, new_dof_id);
               }
            }

            // Replace in all boundary elements with material mat.
            int dof_bdr_cnt = 0;
            if (dof < n_bdr_dofs)
            {
               dof_bdre.GetRow(dof, dof_boundaries);
               dof_bdr_cnt = dof_boundaries.Size();
            }
            for (int b = 0; b < dof_bdr_cnt; b++)
            {
               int face_id = pmesh.GetBdrFace(dof_boundaries[b]);
               int elem_id, tmp;
               pmesh.GetFaceElements(face_id, &elem_id, &tmp);
               if (pmesh.GetAttribute(elem_id) == *mat)
               {
                  std::cout << "Replacing DOF (for boundary): "
                            << dof << " -> " << new_dof_id
                            << " in BE " << dof_boundaries[b] << std::endl;
                  new_bdre_dof.ReplaceConnection(dof_boundaries[b],
                                                 dof, new_dof_id);
               }
            }

            // TODO go over faces (in face_dof) that have the replaced dof (the
            // old id), and check if they have the higher el-attributes on
            // noth sides. For such faces, the face_dof table should be updated
            // with the  new_dof_id.
            // These are faces that touch the interface at a point or an edge.

            ndofs++;
            mat++;
         }
      }

      // Used only for visualization.
      // Must be visualized before the space update.
      x_vis(dof) = dof_mat_cnt;
   }

   // Send the solution by socket to a GLVis server.
   if (vis)
   {
      int size = 350;
      char vishost[] = "localhost";
      int  visport   = 19916;
      const int myid = pfes.GetMyRank(), num_procs = pfes.GetNRanks();

      socketstream sol_sock_x(vishost, visport);
      sol_sock_x << "parallel " << num_procs << " " << myid << "\n";
      sol_sock_x.precision(8);
      sol_sock_x << "solution\n" << pmesh << x_vis;
      sol_sock_x << "window_geometry " << 0 << " " << 0 << " "
                                       << size << " " << size << "\n"
                 << "window_title '" << "Duplicated DOFs" << "'\n"
                 << "keys mRjlc\n" << flush;
   }

   if (print)
   {
      PrintDofElemTable(new_elem_dof, pmesh, 0, false);
   }

   // Remove face dofs for cut faces.
   const Table &face_dof = pfes.GetFaceToDofTable();
   Table new_face_dof(face_dof);
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      auto *ftr = pmesh.GetFaceElementTransformations(f, 3);
      if (ftr->Elem2No > 0 &&
          pmesh.GetAttribute(ftr->Elem1No) != pmesh.GetAttribute(ftr->Elem2No))
      {
         if (print)
         {
            std::cout << ftr->Elem1No << " " << ftr->Elem2No << std::endl;
            std::cout << pmesh.GetAttribute(ftr->Elem1No) << " "
                      << pmesh.GetAttribute(ftr->Elem2No) << std::endl;
            std::cout << "Removing face dofs for face " << f << std::endl;
         }
         new_face_dof.RemoveRow(f);
      }
   }
   new_face_dof.Finalize();

   // Cut the space.
   pfes.ReplaceElemDofTable(new_elem_dof, ndofs);
   pfes.ReplaceBdrElemDofTable(new_bdre_dof);
   pfes.ReplaceFaceDofTable(new_face_dof);
}

void MeshUpdate(ParGridFunction &dx_dt, const ParGridFunction &v)
{
   VectorGridFunctionCoefficient v_coeff(&v);
   dx_dt.ProjectDiscCoefficient(v_coeff, GridFunction::ARITHMETIC);
}

void VisualizeL2(socketstream &sock, ParGridFunction &gf,
                 int size, int x, int y)
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   ParMesh *pmesh = gf.ParFESpace()->GetParMesh();
   const int order = gf.ParFESpace()->GetOrder(0);
   L2_FECollection fec(order, pmesh->Dimension());
   ParFiniteElementSpace pfes(pmesh, &fec, gf.ParFESpace()->GetVDim());
   ParGridFunction gf_l2(&pfes);
   gf_l2.ProjectGridFunction(gf);

   char vishost[] = "localhost";
   int  visport   = 19916;

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh->PrintAsOne(sock);
      gf_l2.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys = (gf_l2.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjl" : "mmaaAcl";

         sock << "window_title '" << "Velocity" << "'\n"
              << "window_geometry "
              << x << " " << y << " " << size << " " << size << "\n"
              << "keys " << keys;
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
   }
   while (connection_failed);
}


} // namespace hydrodynamics

} // namespace mfem
