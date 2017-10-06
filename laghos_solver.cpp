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

#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem {
  Timer::Timer(const std::string &name_) :
    name(name_),
    disabled(false),
    startTime(0),
    timeTaken(0),
    iterations(0),
    dofs(0) {}

  void Timer::disable() {
    disabled = true;
  }

  void Timer::enable() {
    disabled = false;
  }

  void Timer::tic() {
    startTime = occa::sys::currentTime();
  }

  void Timer::toc() {
    toc(occa::getDevice());
  }

  void Timer::toc(occa::device &device) {
    device.finish();
    const double endTime = occa::sys::currentTime();
    timeTaken += (endTime - startTime);
    ++iterations;
  }

  void Timer::addDofs(const long dofs_) {
    dofs += dofs_;
  }

  void Timer::print(const int nameFieldLength) {
    if (!disabled) {
      int mpi_rank, num_procs;
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

      double maxTimeTaken;
      double totalIterations;
      double totalDofs;
      if (num_procs > 1) {
        MPI_Reduce(&timeTaken , &maxTimeTaken   , 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&iterations, &totalIterations, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&dofs      , &totalDofs      , 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        maxTimeTaken = timeTaken;
        totalIterations = iterations;
        totalDofs = dofs;
      }
      if (mpi_rank == 0) {
        std::string tab(nameFieldLength - (int) name.size(), ' ');
        std::cout << '\n'
                  << name << tab << " | Time Taken       | " << maxTimeTaken << '\n'
                  << name << tab << " | Iterations       | " << totalIterations << '\n'
                  << name << tab << " | Iteration / Time | " << (totalIterations / maxTimeTaken) << '\n'
                  << name << tab << " | MDofs / Time     | " << (1e-6 * totalDofs / maxTimeTaken) << '\n';
      }
    }
  }


  HydroTimers::HydroTimers() :
    mult("LagrangianHydroOperator Mult"),
    cgH1("CG(H1)"),
    cgL2("CG(L2)"),
    force("Force Mult"),
    forceT("Force Mult Transpose"),
    quadratureData("Update Quadrature Data") {}

  void HydroTimers::disable() {
    mult.disable();
    cgH1.disable();
    cgL2.disable();
    force.disable();
    forceT.disable();
    quadratureData.disable();
  }

  void HydroTimers::enable() {
    mult.enable();
    cgH1.enable();
    cgL2.enable();
    force.enable();
    forceT.enable();
    quadratureData.enable();
  }

  void HydroTimers::print() {
    int maxNameLength = 0;
    std::string names[6] = {mult.name,
                           cgH1.name, cgL2.name,
                           force.name, forceT.name,
                           quadratureData.name};
    for (int i = 0; i < 6; ++i) {
      maxNameLength = std::max<int>(maxNameLength, names[i].size());
    }

    mult.print(maxNameLength);
    cgH1.print(maxNameLength);
    cgL2.print(maxNameLength);
    force.print(maxNameLength);
    forceT.print(maxNameLength);
    quadratureData.print(maxNameLength);
  }


  namespace miniapps {
    void VisualizeField(socketstream &sock, const char *vishost, int visport,
                        ParGridFunction &gf, const char *title,
                        int x, int y, int w, int h, bool vec) {
      ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
      MPI_Comm comm = pmesh.GetComm();

      int num_procs, myid;
      MPI_Comm_size(comm, &num_procs);
      MPI_Comm_rank(comm, &myid);

      bool newly_opened = false;
      int connection_failed;

      do {
        if (myid == 0) {
          if (!sock.is_open() || !sock) {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
          }
          sock << "solution\n";
        }

        pmesh.PrintAsOne(sock);
        gf.SaveAsOne(sock);

        if (myid == 0 && newly_opened) {
          sock << "window_title '" << title << "'\n"
               << "window_geometry "
               << x << " " << y << " " << w << " " << h << "\n"
               << "keys maaAc";
          if ( vec ) { sock << "vvv"; }
          sock << endl;
        }

        if (myid == 0) {
          connection_failed = !sock && !newly_opened;
        }
        MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
      }
      while (connection_failed);
    }

  } // namespace miniapps

  namespace hydrodynamics {
    LagrangianHydroOperator::LagrangianHydroOperator(Problem problem_,
                                                     OccaFiniteElementSpace &o_H1FESpace_,
                                                     OccaFiniteElementSpace &o_L2FESpace_,
                                                     Array<int> &ess_tdofs_,
                                                     OccaGridFunction &rho0,
                                                     double cfl_,
                                                     double gamma_,
                                                     bool use_viscosity_)
    : TimeDependentOperator(o_L2FESpace_.GetVSize() + 2*o_H1FESpace_.GetVSize()),
      problem(problem_),
      device(o_H1FESpace_.GetDevice()),
      o_H1FESpace(o_H1FESpace_),
      o_L2FESpace(o_L2FESpace_),
      o_H1compFESpace(o_H1FESpace.GetMesh(),
                      o_H1FESpace.FEColl(),
                      1),
      H1FESpace(*((ParFiniteElementSpace*) o_H1FESpace.GetFESpace())),
      L2FESpace(*((ParFiniteElementSpace*) o_L2FESpace.GetFESpace())),
      ess_tdofs(ess_tdofs_),
      dim(H1FESpace.GetMesh()->Dimension()),
      elements(H1FESpace.GetMesh()->GetNE()),
      l2dofs_cnt(L2FESpace.GetFE(0)->GetDof()),
      h1dofs_cnt(H1FESpace.GetFE(0)->GetDof()),
      cfl(cfl_),
      gamma(gamma_),
      use_viscosity(use_viscosity_),
      Mv(&H1FESpace),
      Me_inv(l2dofs_cnt, l2dofs_cnt, elements),
      integ_rule(IntRules.Get(H1FESpace.GetMesh()->GetElementBaseGeometry(),
                              3*H1FESpace.GetOrder(0) + L2FESpace.GetOrder(0) - 1)),
      quad_data(dim, elements, integ_rule.GetNPoints()),
      quad_data_is_current(false),
      VMass(o_H1compFESpace, integ_rule, &quad_data),
      EMass(o_L2FESpace, integ_rule, &quad_data),
      Force(o_H1FESpace, o_L2FESpace, integ_rule, &quad_data),
      timers() {

      Vector rho0_ = rho0;
      GridFunction rho0_gf(&L2FESpace, rho0_.GetData());
      GridFunctionCoefficient rho_coeff(&rho0_gf);

      // Standard local assembly and inversion for energy mass matrices.
      DenseMatrix Me(l2dofs_cnt);
      DenseMatrixInverse inv(&Me);
      MassIntegrator mi(rho_coeff, &integ_rule);
      for (int el = 0; el < elements; ++el) {
        mi.AssembleElementMatrix(*L2FESpace.GetFE(el),
                                 *L2FESpace.GetElementTransformation(el), Me);
        inv.Factor();
        inv.GetInverseMatrix(Me_inv(el));
      }

      // Standard assembly for the velocity mass matrix.
      VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
      Mv.AddDomainIntegrator(vmi);
      Mv.Assemble();

      // Initial local mesh size (assumes similar cells).
      double loc_area = 0.0, glob_area;
      int glob_z_cnt;
      ParMesh *pm = H1FESpace.GetParMesh();
      for (int el = 0; el < elements; ++el) {
        loc_area += pm->GetElementVolume(el);
      }
      MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
      MPI_Allreduce(&elements, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
      switch (pm->GetElementBaseGeometry(0)) {
      case Geometry::SQUARE:
        quad_data.h0 = sqrt(glob_area / glob_z_cnt);
        break;
      case Geometry::TRIANGLE:
        quad_data.h0 = sqrt(2.0 * glob_area / glob_z_cnt);
        break;
      case Geometry::CUBE:
        quad_data.h0 = pow(glob_area / glob_z_cnt, 1.0/3.0);
        break;
      case Geometry::TETRAHEDRON:
        quad_data.h0 = pow(6.0 * glob_area / glob_z_cnt, 1.0/3.0);
        break;
      default: MFEM_ABORT("Unknown zone type!");
      }
      quad_data.h0 /= (double) H1FESpace.GetOrder(0);

      // Setup OCCA QuadratureData
      quad_data.device = device;

      quad_data.dqMaps = OccaDofQuadMaps::Get(device,
                                              o_H1FESpace,
                                              integ_rule);
      quad_data.geom = OccaGeometry::Get(device,
                                         o_H1FESpace,
                                         integ_rule);

      quad_data.Jac0inv = quad_data.geom.invJ;

      OccaVector rhoValues;
      rho0.ToQuad(integ_rule, rhoValues);

      SetProperties(o_H1FESpace, integ_rule, quad_data.props);
      quad_data.props["defines/H0"]            = quad_data.h0;
      quad_data.props["defines/GAMMA"]         = gamma;
      quad_data.props["defines/CFL"]           = cfl;
      quad_data.props["defines/USE_VISCOSITY"] = use_viscosity;

      occa::kernel initKernel = device.buildKernel("occa://laghos/quadratureData.okl",
                                                   "InitQuadratureData",
                                                   quad_data.props);

      initKernel(elements,
                 rhoValues,
                 quad_data.geom.detJ,
                 quad_data.dqMaps.quadWeights,
                 quad_data.rho0DetJ0w);

      updateKernel = device.buildKernel("occa://laghos/quadratureData.okl",
                                        stringWithDim("UpdateQuadratureData", dim),
                                        quad_data.props);

      // Needs quad_data.rho0DetJ0w
      Force.Setup();
      VMass.Setup();
      EMass.Setup();

      cg_print_level = 0;
      cg_max_iters   = 200;
      cg_rel_tol     = 1e-16;
      cg_abs_tol     = 0;
    }

    void LagrangianHydroOperator::Mult(const OccaVector &S, OccaVector &dS_dt) const {
      timers.mult.tic();
      dS_dt = 0.0;

      // Make sure that the mesh positions correspond to the ones in S. This is
      // needed only because some mfem time integrators don't update the solution
      // vector at every intermediate stage (hence they don't change the mesh).
      const int Vsize_h1 = H1FESpace.GetVSize();
      const int Vsize_l2 = L2FESpace.GetVSize();

      // The monolithic BlockVector stores the unknown fields as follows:
      // - Position
      // - Velocity
      // - Specific Internal Energy
      OccaVector x = S.GetRange(0         , Vsize_h1);
      OccaVector v = S.GetRange(Vsize_h1  , Vsize_h1);
      OccaVector e = S.GetRange(2*Vsize_h1, Vsize_l2);

      OccaVector dx = dS_dt.GetRange(0         , Vsize_h1);
      OccaVector dv = dS_dt.GetRange(Vsize_h1  , Vsize_h1);
      OccaVector de = dS_dt.GetRange(2*Vsize_h1, Vsize_l2);

      Vector h_x = x;
      ParGridFunction h_px(&H1FESpace, h_x.GetData());

      o_H1FESpace.GetMesh()->NewNodes(h_px, false);
      UpdateQuadratureData(S);

      // Set dx_dt = v (explicit).
      dx = v;

      // Solve for velocity.
      OccaVector one(Vsize_l2);
      OccaVector rhs(Vsize_h1);
      one = 1.0;

      timers.force.tic();
      Force.Mult(one, rhs);
      timers.force.toc();
      timers.force.addDofs(o_H1FESpace.GetGlobalTrueVSize());
      rhs.Neg();

      OccaVector B(o_H1compFESpace.GetTrueVSize());
      OccaVector X(o_H1compFESpace.GetTrueVSize());

      // Partial assembly solve for each velocity component.
      dv = 0.0;

      for (int c = 0; c < dim; c++) {
        const int size = o_H1compFESpace.GetVSize();
        OccaVector rhs_c = rhs.GetRange(c*size, size);
        OccaVector dv_c  = dv.GetRange(c*size, size);

        // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
        // we must enforce v_x/y/z = 0 for the velocity components.
        Array<int> ess_bdr(H1FESpace.GetParMesh()->bdr_attributes.Max());
        ess_bdr = 0;
        ess_bdr[c] = 1;

        o_H1compFESpace.GetProlongationOperator()->MultTranspose(rhs_c, B);
        o_H1compFESpace.GetRestrictionOperator()->Mult(dv_c, X);

        // True dofs as if there's only one component.
        Array<int> c_tdofs;
        o_H1compFESpace.GetFESpace()->GetEssentialTrueDofs(ess_bdr, c_tdofs);

        VMass.SetEssentialTrueDofs(c_tdofs);
        VMass.EliminateRHS(B);

        {
          TCGSolver<OccaVector> cg(H1FESpace.GetParMesh()->GetComm());
          cg.SetOperator(VMass);
          cg.SetRelTol(sqrt(cg_rel_tol));
          cg.SetAbsTol(sqrt(cg_abs_tol));
          cg.SetMaxIter(cg_max_iters);
          cg.SetPrintLevel(cg_print_level);

          timers.cgH1.tic();
          cg.Mult(B, X);
          timers.cgH1.toc();
          timers.cgH1.addDofs(cg.GetNumIterations()
                              * o_H1compFESpace.GetGlobalTrueVSize());
        }

        o_H1compFESpace.GetProlongationOperator()->Mult(X, dv_c);
      }

      // Solve for energy, assemble the energy source if such exists.
      LinearForm *e_source = NULL;
      if ((problem == vortex) &&
          (dim == 2)) {
        e_source = new LinearForm(&L2FESpace);
        TaylorCoefficient coeff;
        DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
        e_source->AddDomainIntegrator(d);
        e_source->Assemble();
      }

      OccaVector forceRHS(Vsize_l2);
      timers.forceT.tic();
      Force.MultTranspose(v, forceRHS);
      timers.forceT.toc();
      timers.forceT.addDofs(o_L2FESpace.GetGlobalTrueVSize());

      if (e_source) {
        forceRHS += *e_source;
      }

      {
        TCGSolver<OccaVector> cg(L2FESpace.GetParMesh()->GetComm());
        cg.SetOperator(EMass);
        cg.SetRelTol(sqrt(cg_rel_tol));
        cg.SetAbsTol(sqrt(cg_abs_tol));
        cg.SetMaxIter(cg_max_iters);
        cg.SetPrintLevel(cg_print_level);

        timers.cgL2.tic();
        cg.Mult(forceRHS, de);
        timers.cgL2.toc();
        timers.cgL2.addDofs(cg.GetNumIterations()
                            * L2FESpace.GlobalTrueVSize());
      }

      delete e_source;
      quad_data_is_current = false;
      timers.mult.toc();
    }

    double LagrangianHydroOperator::GetTimeStepEstimate(const OccaVector &S) const {
      OccaVector x = S.GetRange(0, H1FESpace.GetVSize());
      Vector h_x = x;
      ParGridFunction h_px(&H1FESpace, h_x.GetData());
      o_H1FESpace.GetMesh()->NewNodes(h_px, false);

      UpdateQuadratureData(S);

      double glob_dt_est;
      MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                    H1FESpace.GetParMesh()->GetComm());
      return glob_dt_est;
    }

    void LagrangianHydroOperator::ResetTimeStepEstimate() const {
      quad_data.dt_est = numeric_limits<double>::infinity();
    }

    void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) {
      rho.SetSpace(&L2FESpace);

      DenseMatrix Mrho(l2dofs_cnt);
      Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
      Array<int> dofs(l2dofs_cnt);
      DenseMatrixInverse inv(&Mrho);
      MassIntegrator mi(&integ_rule);
      DensityIntegrator di(quad_data);

      Vector rho0DetJ0w = quad_data.rho0DetJ0w;

      for (int el = 0; el < elements; ++el) {
        di.AssembleRHSElementVect(*L2FESpace.GetFE(el),
                                  *L2FESpace.GetElementTransformation(el),
                                  integ_rule,
                                  rho0DetJ0w,
                                  rhs);
        mi.AssembleElementMatrix(*L2FESpace.GetFE(el),
                                 *L2FESpace.GetElementTransformation(el),
                                 Mrho);
        inv.Factor();
        inv.Mult(rhs, rho_z);
        L2FESpace.GetElementDofs(el, dofs);
        rho.SetSubVector(dofs, rho_z);
      }
    }

    LagrangianHydroOperator::~LagrangianHydroOperator() {}

    void LagrangianHydroOperator::UpdateQuadratureData(const OccaVector &S) const {
      if (quad_data_is_current) {
        return;
      }

      quad_data_is_current = true;

      const int vSize = o_H1FESpace.GetVSize();
      const int eSize = o_L2FESpace.GetVSize();

      OccaGridFunction v(&o_H1FESpace, S.GetRange(vSize  , vSize));
      OccaGridFunction e(&o_L2FESpace, S.GetRange(2*vSize, eSize));

      quad_data.geom = OccaGeometry::Get(device,
                                         o_H1FESpace,
                                         integ_rule);

      timers.quadratureData.tic();
      OccaVector v2(device,
                    o_H1FESpace.GetVDim() * o_H1FESpace.GetLocalDofs() * elements);
      o_H1FESpace.GlobalToLocal(v, v2);

      OccaVector eValues;
      e.ToQuad(integ_rule, eValues);

      updateKernel(elements,
                   quad_data.dqMaps.dofToQuad,
                   quad_data.dqMaps.dofToQuadD,
                   quad_data.dqMaps.quadWeights,
                   v2,
                   eValues,
                   quad_data.rho0DetJ0w,
                   quad_data.Jac0inv,
                   quad_data.geom.J,
                   quad_data.geom.invJ,
                   quad_data.geom.detJ,
                   quad_data.stressJinvT,
                   quad_data.dtEst);
      timers.quadratureData.toc();
      timers.quadratureData.addDofs(o_H1FESpace.GetMesh()->GetNE() *
                                    integ_rule.GetNPoints());

      quad_data.dt_est = quad_data.dtEst.Min();
    }
  } // namespace hydrodynamics
} // namespace mfem

#endif // MFEM_USE_MPI
