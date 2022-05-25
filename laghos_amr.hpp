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

enum estimator: int { custom = 0, jjt = 1, zz = 2, kelly = 3 };

class AMREstimatorIntegrator: public DiffusionIntegrator
{
   int NE, e;
   ParMesh *pmesh;
   enum class mode { diffusion, one, two };
   const mode flux_mode;
   ConstantCoefficient one {1.0};
   const int max_level;
   const double jac_threshold;
public:
   AMREstimatorIntegrator(ParMesh *pmesh,
                          const int max_level,
                          const double jac_threshold,
                          const mode flux_mode = mode::two):
      DiffusionIntegrator(one),
      NE(pmesh->GetNE()),
      pmesh(pmesh),
      flux_mode(flux_mode),
      max_level(max_level),
      jac_threshold(jac_threshold) { }

   void Reset() { e = 0; NE = pmesh->GetNE(); }

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
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef = true,
                                   const IntegrationRule *ir = NULL);
private:
   void ComputeElementFlux1(const FiniteElement &el,
                            ElementTransformation &Trans,
                            const Vector &u,
                            const FiniteElement &fluxelem,
                            Vector &flux);

   void ComputeElementFlux2(const int e,
                            const FiniteElement &el,
                            ElementTransformation &Trans,
                            const FiniteElement &fluxelem,
                            Vector &flux);
};

// AMR operator
class AMR
{
   const int order = 3; // should be computed
   ParMesh *pmesh;
   const int myid, dim, sdim;

   L2_FECollection flux_fec;
   ParFiniteElementSpace flux_fes;

   RT_FECollection *smooth_flux_fec = nullptr;
   ErrorEstimator *estimator = nullptr;
   ThresholdRefiner *refiner = nullptr;
   ThresholdDerefiner *derefiner = nullptr;
   AMREstimatorIntegrator *integ = nullptr;

   const struct Options
   {
      int estimator;
      double ref_threshold;
      double jac_threshold;
      double deref_threshold;
      int max_level;
      int nc_limit;
      double blast_size;
      double blast_energy;
      Vertex blast_position;
   } opt;

public:
   AMR(ParMesh *pmesh,
       int estimator,
       double ref_t, double jac_t, double deref_t,
       int max_level, int nc_limit,
       double blast_size, double blast_energy, double *blast_position);

   ~AMR();

   void Setup(ParGridFunction&);

   void Reset();

   void Update(hydrodynamics::LagrangianHydroOperator &hydro,
               ODESolver *ode_solver,
               BlockVector &S,
               BlockVector &S_old,
               ParGridFunction &x,
               ParGridFunction &v,
               ParGridFunction &e,
               ParGridFunction &m,
               Array<int> &true_offset,
               const int bdr_attr_max,
               Array<int> &ess_tdofs,
               Array<int> &ess_vdofs);
};

} // namespace amr

} // namespace mfem

#endif // MFEM_LAGHOS_AMR
