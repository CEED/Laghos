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

#ifndef MFEM_LAGHOS_AMR
#define MFEM_LAGHOS_AMR

#include "mfem.hpp"
#include "laghos_solver.hpp"

namespace mfem
{

namespace amr
{

enum estimator: int { std = 0, rho = 1, zz = 2, kelly = 3 };


static void Update(BlockVector &S, BlockVector &S_tmp,
                   Array<int> &true_offset,
                   ParGridFunction &x_gf,
                   ParGridFunction &v_gf,
                   ParGridFunction &e_gf,
                   ParGridFunction &m_gf)
{
   ParFiniteElementSpace* H1FESpace = x_gf.ParFESpace();
   ParFiniteElementSpace* L2FESpace = e_gf.ParFESpace();
   ParFiniteElementSpace* MEFESpace = m_gf.ParFESpace();

   H1FESpace->Update();
   L2FESpace->Update();
   MEFESpace->Update();

   const int Vsize_h1 = H1FESpace->GetVSize();
   const int Vsize_l2 = L2FESpace->GetVSize();

   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_h1;
   true_offset[2] = true_offset[1] + Vsize_h1;
   true_offset[3] = true_offset[2] + Vsize_l2;

   S_tmp = S;
   S.Update(true_offset);
   const Operator* H1Update = H1FESpace->GetUpdateOperator();
   const Operator* L2Update = L2FESpace->GetUpdateOperator();
   H1Update->Mult(S_tmp.GetBlock(0), S.GetBlock(0));
   H1Update->Mult(S_tmp.GetBlock(1), S.GetBlock(1));
   L2Update->Mult(S_tmp.GetBlock(2), S.GetBlock(2));

   x_gf.MakeRef(H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(L2FESpace, S, true_offset[2]);
   x_gf.SyncAliasMemory(S);
   v_gf.SyncAliasMemory(S);
   e_gf.SyncAliasMemory(S);

   S_tmp.Update(true_offset);
   m_gf.Update();

   //H1FESpace->UpdatesFinished();
   //L2FESpace->UpdatesFinished();
   //MEFESpace->UpdatesFinished();
}

static void FindElementsWithVertex(const Mesh* mesh, const Vertex &vert,
                                   const double size, Array<int> &elements)
{
   Array<int> v;

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      mesh->GetElementVertices(i, v);
      for (int j = 0; j < v.Size(); j++)
      {
         double dist = 0.0;
         for (int l = 0; l < mesh->SpaceDimension(); l++)
         {
            double d = vert(l) - mesh->GetVertex(v[j])[l];
            dist += d*d;
         }
         if (dist <= size*size) { elements.Append(i); break; }
      }
   }
}

static void Pow(Vector &vec, double p)
{
   for (int i = 0; i < vec.Size(); i++)
   {
      vec(i) = std::pow(vec(i), p);
   }
}

static void GetPerElementMinMax(const GridFunction &gf,
                                Vector &elem_min, Vector &elem_max,
                                int int_order = -1)
{
   const FiniteElementSpace *space = gf.FESpace();
   int ne = space->GetNE();

   if (int_order < 0) { int_order = space->GetOrder(0) + 1; }

   elem_min.SetSize(ne);
   elem_max.SetSize(ne);

   Vector vals, tmp;
   for (int i = 0; i < ne; i++)
   {
      int geom = space->GetFE(i)->GetGeomType();
      const IntegrationRule &ir = IntRules.Get(geom, int_order);

      gf.GetValues(i, ir, vals);

      if (space->GetVDim() > 1)
      {
         Pow(vals, 2.0);
         for (int vd = 1; vd < space->GetVDim(); vd++)
         {
            gf.GetValues(i, ir, tmp, vd+1);
            Pow(tmp, 2.0);
            vals += tmp;
         }
         Pow(vals, 0.5);
      }

      elem_min(i) = vals.Min();
      elem_max(i) = vals.Max();
   }
}

class EstimatorIntegrator: public DiffusionIntegrator
{
   enum class mode { diffusion, one, two };
   const mode flux_mode;
   ConstantCoefficient one {1.0};
public:
   EstimatorIntegrator(const mode flux_mode = mode::diffusion):
      DiffusionIntegrator(one), flux_mode(flux_mode) { }

   double ComputeFluxEnergy(const FiniteElement &fluxelem,
                            ElementTransformation &Trans,
                            Vector &flux, Vector *d_energy = NULL)
   {
      if (flux_mode == mode::diffusion)
      {
         return DiffusionIntegrator::ComputeFluxEnergy(fluxelem, Trans, flux, d_energy);
      }
      // Not implemented for other modes
      MFEM_ABORT("Not implemented!");
      return 0.0;
   }

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux,
                                   bool with_coef = false);
};

class Operator
{
   const int order = 3;
   ParMesh *pmesh;
   const int myid, dim, sdim;
   const struct Options
   {
      int estimator;
      double ref_threshold;
      double jac_threshold;
      double deref_threshold;
      int max_level;
      double blast_size;
      int nc_limit;
      double blast_energy;
      Vertex blast_position;
   } opt;

   // AMR estimator setup
   L2_FECollection *flux_fec;
   RT_FECollection *smooth_flux_fec;

   ErrorEstimator *estimator = nullptr;
   ThresholdRefiner *refiner = nullptr;
   ThresholdDerefiner *derefiner = nullptr;
   amr::EstimatorIntegrator *ei = nullptr;

public:
   Operator(ParMesh *pmesh,
            int estimator,
            double ref_t, double jac_t, double deref_t,
            int max_level,
            const double blast_size, const int nc_limit,
            const double blast_energy, const double *blast_position);

   ~Operator();

   void Setup(ParGridFunction&);

   void Reset();

   void Update(BlockVector &S,
               BlockVector &S_old,
               ParGridFunction &x,
               ParGridFunction &v,
               ParGridFunction &e,
               ParGridFunction &m,
               Array<int> &true_offset,
               hydrodynamics::LagrangianHydroOperator&,
               ODESolver*,
               const int bdr_attr_max,
               Array<int> &ess_tdofs,
               Array<int> &ess_vdofs);
};

} // namespace amr

} // namespace mfem

#endif // MFEM_LAGHOS_AMR
