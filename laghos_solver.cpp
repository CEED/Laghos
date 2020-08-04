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

namespace mfem
{

namespace hydrodynamics
{

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
    ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
    MPI_Comm comm = pmesh.GetComm();

    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

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

        pmesh.PrintAsOne(sock);
        gf.SaveAsOne(sock);

        if (myid == 0 && newly_opened)
        {
            const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                               ? "mAcRjlPPPPPPPP" : "maaAcl";

            sock << "window_title '" << title << "'\n"
                 << "window_geometry "
                 << x << " " << y << " " << w << " " << h << "\n"
                 << "keys " << keys;
            if ( vec ) {
                sock << "vvv";
            }
            sock << endl;
        }

        if (myid == 0)
        {
            connection_failed = !sock && !newly_opened;
        }
        MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
    }
    while (connection_failed);
}

LagrangianHydroOperator::LagrangianHydroOperator(int size,
        ParFiniteElementSpace &h1_fes,
        ParFiniteElementSpace &l2_fes,
        Array<int> &essential_tdofs,
        ParGridFunction &rho0,
        int source_type_, double cfl_,
        Coefficient *material_,
        bool visc, bool pa,
        double cgt, int cgiter,
        double ftz,
        int h1_basis_type,
        bool noMvSolve_,
        bool noMeSolve_)
    : TimeDependentOperator(size),
      H1FESpace(h1_fes), L2FESpace(l2_fes),
      ess_tdofs(essential_tdofs),
      dim(h1_fes.GetMesh()->Dimension()),
      nzones(h1_fes.GetMesh()->GetNE()),
      l2dofs_cnt(l2_fes.GetFE(0)->GetDof()),
      h1dofs_cnt(h1_fes.GetFE(0)->GetDof()),
      source_type(source_type_), cfl(cfl_),
      use_viscosity(visc), p_assembly(pa), cg_rel_tol(cgt), cg_max_iter(cgiter),
      ftz_tol(ftz),
      noMvSolve(noMvSolve_), noMeSolve(noMeSolve_),
      material_pcf(material_),
      Mv(&h1_fes), Mv_spmat_copy(),
      Me(l2dofs_cnt, l2dofs_cnt, nzones), Me_inv(l2dofs_cnt, l2dofs_cnt, nzones),
      integ_rule(IntRules.Get(h1_fes.GetMesh()->GetElementBaseGeometry(0),
                              3*h1_fes.GetOrder(0) + l2_fes.GetOrder(0) - 1)),
      quad_data(dim, nzones, integ_rule.GetNPoints()),
      quad_data_is_current(false), forcemat_is_assembled(false),
      tensors1D(H1FESpace.GetFE(0)->GetOrder(), L2FESpace.GetFE(0)->GetOrder(),
                int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))),
                h1_basis_type == BasisType::Positive),
      evaluator(H1FESpace, &tensors1D),
      Force(&l2_fes, &h1_fes), ForcePA(&quad_data, h1_fes, l2_fes, &tensors1D),
      VMassPA(&quad_data, H1FESpace, &tensors1D), VMassPA_prec(H1FESpace),
      locEMassPA(&quad_data, l2_fes, &tensors1D),
      locCG(), timer()
{

    GridFunctionCoefficient rho_coeff(&rho0);

    // Standard local assembly and inversion for energy mass matrices.
    MassIntegrator mi(rho_coeff, &integ_rule);
    for (int i = 0; i < nzones; i++)
    {
        DenseMatrixInverse inv(&Me(i));
        mi.AssembleElementMatrix(*l2_fes.GetFE(i),
                                 *l2_fes.GetElementTransformation(i), Me(i));
        inv.Factor();
        inv.GetInverseMatrix(Me_inv(i));
    }

    // Standard assembly for the velocity mass matrix.
    VectorMassIntegrator *vmi = new VectorMassIntegrator(rho_coeff, &integ_rule);
    Mv.AddDomainIntegrator(vmi);
    Mv.Assemble();
    Mv_spmat_copy = Mv.SpMat();

    // Values of rho0DetJ0 and Jac0inv at all quadrature points.
    const int nqp = integ_rule.GetNPoints();
    Vector rho_vals(nqp);
    for (int i = 0; i < nzones; i++)
    {
        rho0.GetValues(i, integ_rule, rho_vals);
        ElementTransformation *T = h1_fes.GetElementTransformation(i);
        for (int q = 0; q < nqp; q++)
        {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);

            DenseMatrixInverse Jinv(T->Jacobian());
            Jinv.GetInverseMatrix(quad_data.Jac0inv(i*nqp + q));

            const double rho0DetJ0 = T->Weight() * rho_vals(q);
            quad_data.rho0DetJ0w(i*nqp + q) = rho0DetJ0 *
                                              integ_rule.IntPoint(q).weight;
        }
    }

    // Initial local mesh size (assumes all mesh elements are of the same type).
    double loc_area = 0.0, glob_area;
    int loc_z_cnt = nzones, glob_z_cnt;
    ParMesh *pm = H1FESpace.GetParMesh();
    for (int i = 0; i < nzones; i++) {
        loc_area += pm->GetElementVolume(i);
    }
    MPI_Allreduce(&loc_area, &glob_area, 1, MPI_DOUBLE, MPI_SUM, pm->GetComm());
    MPI_Allreduce(&loc_z_cnt, &glob_z_cnt, 1, MPI_INT, MPI_SUM, pm->GetComm());
    switch (pm->GetElementBaseGeometry(0))
    {
    case Geometry::SEGMENT:
        quad_data.h0 = glob_area / glob_z_cnt;
        break;
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
    default:
        MFEM_ABORT("Unknown zone type!");
    }
    quad_data.h0 /= (double) H1FESpace.GetOrder(0);

    if (p_assembly)
    {
        // Setup the preconditioner of the velocity mass operator.
        Vector d;
        (dim == 2) ? VMassPA.ComputeDiagonal2D(d) : VMassPA.ComputeDiagonal3D(d);
        VMassPA_prec.SetDiagonal(d);
    }
    else
    {
        ForceIntegrator *fi = new ForceIntegrator(quad_data);
        fi->SetIntRule(&integ_rule);
        Force.AddDomainIntegrator(fi);
        // Make a dummy assembly to figure out the sparsity.
        Force.Assemble(0);
        Force.Finalize(0);
    }

    locCG.SetOperator(locEMassPA);
    locCG.iterative_mode = false;
    locCG.SetRelTol(1e-8);
    locCG.SetAbsTol(1e-8 * numeric_limits<double>::epsilon());
    locCG.SetMaxIter(200);
    locCG.SetPrintLevel(0);
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dS_dt) const
{
    // Make sure that the mesh positions correspond to the ones in S. This is
    // needed only because some mfem time integrators don't update the solution
    // vector at every intermediate stage (hence they don't change the mesh).
    UpdateMesh(S);

    // The monolithic BlockVector stores the unknown fields as follows:
    // (Position, Velocity, Specific Internal Energy).
    Vector* sptr = (Vector*) &S;
    ParGridFunction v;
    const int VsizeH1 = H1FESpace.GetVSize();
    v.MakeRef(&H1FESpace, *sptr, VsizeH1);

    // Set dx_dt = v (explicit).
    ParGridFunction dx;
    dx.MakeRef(&H1FESpace, dS_dt, 0);
    dx = v;

    SolveVelocity(S, dS_dt);
    SolveEnergy(S, v, dS_dt);

    quad_data_is_current = false;
}

void LagrangianHydroOperator::SolveVelocity(const Vector &S,
        Vector &dS_dt) const
{
    UpdateQuadratureData(S);
    AssembleForceMatrix();

    const int VsizeL2 = L2FESpace.GetVSize();
    const int VsizeH1 = H1FESpace.GetVSize();

    // The monolithic BlockVector stores the unknown fields as follows:
    // (Position, Velocity, Specific Internal Energy).
    ParGridFunction dv;
    dv.MakeRef(&H1FESpace, dS_dt, VsizeH1);
    dv = 0.0;

    Vector one(VsizeL2), rhs(VsizeH1), B, X;
    one = 1.0;
    if (p_assembly)
    {
        timer.sw_force.Start();
        ForcePA.Mult(one, rhs);
        if (ftz_tol>0.0)
        {
            for (int i = 0; i < VsizeH1; i++)
            {
                if (fabs(rhs[i]) < ftz_tol)
                {
                    rhs[i] = 0.0;
                }
            }
        }
        timer.sw_force.Stop();
        rhs.Neg();

        if (noMvSolve)
        {
            dv = rhs;

            for (int i=0; i<ess_tdofs.Size(); ++i)
            {
                dv[ess_tdofs[i]] = 0.0;
            }

            return;
        }

        Operator *cVMassPA;
        VMassPA.FormLinearSystem(ess_tdofs, dv, rhs, cVMassPA, X, B);
        CGSolver cg(H1FESpace.GetParMesh()->GetComm());
        cg.SetPreconditioner(VMassPA_prec);
        cg.SetOperator(*cVMassPA);
        cg.SetRelTol(cg_rel_tol);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(cg_max_iter);
        cg.SetPrintLevel(0);
        timer.sw_cgH1.Start();
        cg.Mult(B, X);
        timer.sw_cgH1.Stop();
        timer.H1cg_iter += cg.GetNumIterations();
        VMassPA.RecoverFEMSolution(X, rhs, dv);
        delete cVMassPA;
    }
    else
    {
        timer.sw_force.Start();
        Force.Mult(one, rhs);
        timer.sw_force.Stop();
        rhs.Neg();

        if (noMvSolve)
        {
            dv = rhs;
            for (int i=0; i<ess_tdofs.Size(); ++i)
            {
                dv[ess_tdofs[i]] = 0.0;
            }

            return;
        }

        HypreParMatrix A;
        Mv.FormLinearSystem(ess_tdofs, dv, rhs, A, X, B);
        CGSolver cg(H1FESpace.GetParMesh()->GetComm());
        HypreSmoother prec;
        prec.SetType(HypreSmoother::Jacobi, 1);
        cg.SetPreconditioner(prec);
        cg.SetOperator(A);
        cg.SetRelTol(cg_rel_tol);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(cg_max_iter);
        cg.SetPrintLevel(0);
        timer.sw_cgH1.Start();
        cg.Mult(B, X);
        timer.sw_cgH1.Stop();
        timer.H1cg_iter += cg.GetNumIterations();
        Mv.RecoverFEMSolution(X, rhs, dv);
    }
}

void LagrangianHydroOperator::SolveEnergy(const Vector &S, const Vector &v,
        Vector &dS_dt) const
{
    UpdateQuadratureData(S);
    AssembleForceMatrix();

    const int VsizeL2 = L2FESpace.GetVSize();
    const int VsizeH1 = H1FESpace.GetVSize();

    // The monolithic BlockVector stores the unknown fields as follows:
    // (Position, Velocity, Specific Internal Energy).
    ParGridFunction de;
    de.MakeRef(&L2FESpace, dS_dt, VsizeH1*2);
    de = 0.0;

    // Solve for energy, assemble the energy source if such exists.
    LinearForm *e_source = NULL;
    if (source_type == 1) // 2D Taylor-Green.
    {
        e_source = new LinearForm(&L2FESpace);
        TaylorCoefficient coeff;
        DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
        e_source->AddDomainIntegrator(d);
        e_source->Assemble();
    }
    Array<int> l2dofs;
    Vector e_rhs(VsizeL2), loc_rhs(l2dofs_cnt), loc_de(l2dofs_cnt);
    if (p_assembly)
    {
        timer.sw_force.Start();
        ForcePA.MultTranspose(v, e_rhs);
        timer.sw_force.Stop();

        if (e_source) {
            e_rhs += *e_source;
        }

        if (noMeSolve)
            de = e_rhs;
        else
        {
            for (int z = 0; z < nzones; z++)
            {
                L2FESpace.GetElementDofs(z, l2dofs);
                e_rhs.GetSubVector(l2dofs, loc_rhs);
                locEMassPA.SetZoneId(z);
                timer.sw_cgL2.Start();
                locCG.Mult(loc_rhs, loc_de);
                timer.sw_cgL2.Stop();
                timer.L2dof_iter += locCG.GetNumIterations() * l2dofs_cnt;
                de.SetSubVector(l2dofs, loc_de);
            }
        }
    }
    else
    {
        timer.sw_force.Start();
        Force.MultTranspose(v, e_rhs);
        timer.sw_force.Stop();
        if (e_source) {
            e_rhs += *e_source;
        }

        if (noMeSolve)
            de = e_rhs;
        else
        {
            for (int z = 0; z < nzones; z++)
            {
                L2FESpace.GetElementDofs(z, l2dofs);
                e_rhs.GetSubVector(l2dofs, loc_rhs);
                timer.sw_cgL2.Start();
                Me_inv(z).Mult(loc_rhs, loc_de);
                timer.sw_cgL2.Stop();
                timer.L2dof_iter += l2dofs_cnt;
                de.SetSubVector(l2dofs, loc_de);
            }
        }
    }
    delete e_source;
}

void LagrangianHydroOperator::MultMv(const Vector &u, Vector &v)
{
    if (p_assembly)
    {
        Operator *cVMassPA;
        VMassPA.FormSystemOperator(ess_tdofs, cVMassPA);
        cVMassPA->Mult(u, v);
        delete cVMassPA;
    }
    else
    {
        HypreParMatrix A;
        Mv.FormSystemMatrix(ess_tdofs, A);
        A.Mult(u, v);
    }

    for (int i=0; i<ess_tdofs.Size(); ++i)
    {
        //v[ess_tdofs[i]] = 0.0;
        v[ess_tdofs[i]] = u[ess_tdofs[i]];
    }
}

void LagrangianHydroOperator::MultMe(const Vector &u, Vector &v)
{
    v = 0.0;

    const int VsizeL2 = L2FESpace.GetVSize();

    MFEM_VERIFY(u.Size() == VsizeL2 && u.Size() == v.Size(), "");

    Array<int> l2dofs;
    Vector loc_rhs(l2dofs_cnt), loc_v(l2dofs_cnt);

    if (p_assembly)
    {
        for (int z = 0; z < nzones; z++)
        {
            L2FESpace.GetElementDofs(z, l2dofs);
            u.GetSubVector(l2dofs, loc_rhs);
            locEMassPA.SetZoneId(z);
            locEMassPA.Mult(loc_rhs, loc_v);
            v.SetSubVector(l2dofs, loc_v);
        }
    }
    else
    {
        for (int z = 0; z < nzones; z++)
        {
            L2FESpace.GetElementDofs(z, l2dofs);
            u.GetSubVector(l2dofs, loc_rhs);
            Me(z).Mult(loc_rhs, loc_v);
            v.SetSubVector(l2dofs, loc_v);
        }
    }
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
    Vector* sptr = (Vector*) &S;
    x_gf.MakeRef(&H1FESpace, *sptr, 0);
    H1FESpace.GetParMesh()->NewNodes(x_gf, false);
}

double LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S) const
{
    UpdateMesh(S);
    UpdateQuadratureData(S);

    double glob_dt_est;
    MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                  H1FESpace.GetParMesh()->GetComm());
    return glob_dt_est;
}

void LagrangianHydroOperator::ResetTimeStepEstimate() const
{
    quad_data.dt_est = numeric_limits<double>::infinity();
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho) const
{
    rho.SetSpace(&L2FESpace);

    DenseMatrix Mrho(l2dofs_cnt);
    Vector rhs(l2dofs_cnt), rho_z(l2dofs_cnt);
    Array<int> dofs(l2dofs_cnt);
    DenseMatrixInverse inv(&Mrho);
    MassIntegrator mi(&integ_rule);
    DensityIntegrator di(quad_data);
    di.SetIntRule(&integ_rule);
    for (int i = 0; i < nzones; i++)
    {
        di.AssembleRHSElementVect(*L2FESpace.GetFE(i),
                                  *L2FESpace.GetElementTransformation(i), rhs);
        mi.AssembleElementMatrix(*L2FESpace.GetFE(i),
                                 *L2FESpace.GetElementTransformation(i), Mrho);
        inv.Factor();
        inv.Mult(rhs, rho_z);
        L2FESpace.GetElementDofs(i, dofs);
        rho.SetSubVector(dofs, rho_z);
    }
}

double LagrangianHydroOperator::InternalEnergy(const ParGridFunction &e) const
{
    Vector one(l2dofs_cnt), loc_e(l2dofs_cnt);
    one = 1.0;
    Array<int> l2dofs;

    double loc_ie = 0.0;
    for (int z = 0; z < nzones; z++)
    {
        L2FESpace.GetElementDofs(z, l2dofs);
        e.GetSubVector(l2dofs, loc_e);
        loc_ie += Me(z).InnerProduct(loc_e, one);
    }

    double glob_ie;
    MPI_Allreduce(&loc_ie, &glob_ie, 1, MPI_DOUBLE, MPI_SUM,
                  H1FESpace.GetParMesh()->GetComm());
    return glob_ie;
}

double LagrangianHydroOperator::KineticEnergy(const ParGridFunction &v) const
{
    double loc_ke = 0.5 * Mv_spmat_copy.InnerProduct(v, v);

    double glob_ke;
    MPI_Allreduce(&loc_ke, &glob_ke, 1, MPI_DOUBLE, MPI_SUM,
                  H1FESpace.GetParMesh()->GetComm());
    return glob_ke;
}

void LagrangianHydroOperator::PrintTimingData(bool IamRoot, int steps) const
{
    double my_rt[5], rt_max[5];
    my_rt[0] = timer.sw_cgH1.RealTime();
    my_rt[1] = timer.sw_cgL2.RealTime();
    my_rt[2] = timer.sw_force.RealTime();
    my_rt[3] = timer.sw_qdata.RealTime();
    my_rt[4] = my_rt[0] + my_rt[2] + my_rt[3];
    MPI_Reduce(my_rt, rt_max, 5, MPI_DOUBLE, MPI_MAX, 0, H1FESpace.GetComm());

    HYPRE_Int mydata[2], alldata[2];
    mydata[0] = timer.L2dof_iter;
    mydata[1] = timer.quad_tstep;
    MPI_Reduce(mydata, alldata, 2, HYPRE_MPI_INT, MPI_SUM, 0,
               H1FESpace.GetComm());

    if (IamRoot)
    {
        const HYPRE_Int H1gsize = H1FESpace.GlobalTrueVSize(),
                        L2gsize = L2FESpace.GlobalTrueVSize();
        using namespace std;
        cout << endl;
        cout << "CG (H1) total time: " << rt_max[0] << endl;
        cout << "CG (H1) rate (megadofs x cg_iterations / second): "
             << 1e-6 * H1gsize * timer.H1cg_iter / rt_max[0] << endl;
        cout << endl;
        cout << "CG (L2) total time: " << rt_max[1] << endl;
        cout << "CG (L2) rate (megadofs x cg_iterations / second): "
             << 1e-6 * alldata[0] / rt_max[1] << endl;
        cout << endl;
        // The Force operator is applied twice per time step, on the H1 and the L2
        // vectors, respectively.
        cout << "Forces total time: " << rt_max[2] << endl;
        cout << "Forces rate (megadofs x timesteps / second): "
             << 1e-6 * steps * (H1gsize + L2gsize) / rt_max[2] << endl;
        cout << endl;
        cout << "UpdateQuadData total time: " << rt_max[3] << endl;
        cout << "UpdateQuadData rate (megaquads x timesteps / second): "
             << 1e-6 * alldata[1] * integ_rule.GetNPoints() / rt_max[3] << endl;
        cout << endl;
        cout << "Major kernels total time (seconds): " << rt_max[4] << endl;
        cout << "Major kernels total rate (megadofs x time steps / second): "
             << 1e-6 * steps * (H1gsize + L2gsize) / rt_max[4] << endl;
    }
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
inline double smooth_step_01(double x, double eps)
{
    const double y = (x + eps) / (2.0 * eps);
    if (y < 0.0) {
        return 0.0;
    }
    if (y > 1.0) {
        return 1.0;
    }
    return (3.0 - 2.0 * y) * y * y;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
    if (quad_data_is_current) {
        return;
    }
    timer.sw_qdata.Start();

    const int nqp = integ_rule.GetNPoints();

    ParGridFunction x, v, e;
    Vector* sptr = (Vector*) &S;
    x.MakeRef(&H1FESpace, *sptr, 0);
    v.MakeRef(&H1FESpace, *sptr, H1FESpace.GetVSize());
    e.MakeRef(&L2FESpace, *sptr, 2*H1FESpace.GetVSize());
    Vector e_vals, e_loc(l2dofs_cnt), vector_vals(h1dofs_cnt * dim);
    DenseMatrix Jpi(dim), sgrad_v(dim), Jinv(dim), stress(dim), stressJiT(dim),
                vecvalMat(vector_vals.GetData(), h1dofs_cnt, dim);
    DenseTensor grad_v_ref(dim, dim, nqp);
    Array<int> L2dofs, H1dofs;

    // Batched computations are needed, because hydrodynamic codes usually
    // involve expensive computations of material properties. Although this
    // miniapp uses simple EOS equations, we still want to represent the batched
    // cycle structure.
    int nzones_batch = 3;
    const int nbatches =  nzones / nzones_batch + 1; // +1 for the remainder.
    int nqp_batch = nqp * nzones_batch;
    double *gamma_b = new double[nqp_batch],
    *rho_b = new double[nqp_batch],
    *e_b   = new double[nqp_batch],
    *p_b   = new double[nqp_batch],
    *cs_b  = new double[nqp_batch];
    // Jacobians of reference->physical transformations for all quadrature points
    // in the batch.
    DenseTensor *Jpr_b = new DenseTensor[nzones_batch];
    for (int b = 0; b < nbatches; b++)
    {
        int z_id = b * nzones_batch; // Global index over zones.
        // The last batch might not be full.
        if (z_id == nzones) {
            break;
        }
        else if (z_id + nzones_batch > nzones)
        {
            nzones_batch = nzones - z_id;
            nqp_batch    = nqp * nzones_batch;
        }

        double min_detJ = numeric_limits<double>::infinity();
        for (int z = 0; z < nzones_batch; z++)
        {
            ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
            Jpr_b[z].SetSize(dim, dim, nqp);

            if (p_assembly)
            {
                // Energy values at quadrature point.
                L2FESpace.GetElementDofs(z_id, L2dofs);
                e.GetSubVector(L2dofs, e_loc);
                evaluator.GetL2Values(e_loc, e_vals);

                // All reference->physical Jacobians at the quadrature points.
                H1FESpace.GetElementVDofs(z_id, H1dofs);
                x.GetSubVector(H1dofs, vector_vals);
                evaluator.GetVectorGrad(vecvalMat, Jpr_b[z]);
            }
            else {
                e.GetValues(z_id, integ_rule, e_vals);
            }
            for (int q = 0; q < nqp; q++)
            {
                const IntegrationPoint &ip = integ_rule.IntPoint(q);
                T->SetIntPoint(&ip);
                if (!p_assembly) {
                    Jpr_b[z](q) = T->Jacobian();
                }
                const double detJ = Jpr_b[z](q).Det();
                min_detJ = min(min_detJ, detJ);

                const int idx = z * nqp + q;
                if (material_pcf == NULL) {
                    gamma_b[idx] = 5./3.;    // Ideal gas.
                }
                else {
                    gamma_b[idx] = material_pcf->Eval(*T, ip);
                }
                rho_b[idx] = quad_data.rho0DetJ0w(z_id*nqp + q) / detJ / ip.weight;
                e_b[idx]   = max(0.0, e_vals(q));
            }
            ++z_id;
        }

        // Batched computation of material properties.
        ComputeMaterialProperties(nqp_batch, gamma_b, rho_b, e_b, p_b, cs_b);

        z_id -= nzones_batch;
        for (int z = 0; z < nzones_batch; z++)
        {
            ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
            if (p_assembly)
            {
                // All reference->physical Jacobians at the quadrature points.
                H1FESpace.GetElementVDofs(z_id, H1dofs);
                v.GetSubVector(H1dofs, vector_vals);
                evaluator.GetVectorGrad(vecvalMat, grad_v_ref);
            }
            for (int q = 0; q < nqp; q++)
            {
                const IntegrationPoint &ip = integ_rule.IntPoint(q);
                T->SetIntPoint(&ip);
                // Note that the Jacobian was already computed above. We've chosen
                // not to store the Jacobians for all batched quadrature points.
                const DenseMatrix &Jpr = Jpr_b[z](q);
                CalcInverse(Jpr, Jinv);
                const double detJ = Jpr.Det(), rho = rho_b[z*nqp + q],
                             p = p_b[z*nqp + q], sound_speed = cs_b[z*nqp + q];

                stress = 0.0;
                for (int d = 0; d < dim; d++) {
                    stress(d, d) = -p;
                }

                double visc_coeff = 0.0;
                if (use_viscosity)
                {
                    // Compression-based length scale at the point. The first
                    // eigenvector of the symmetric velocity gradient gives the
                    // direction of maximal compression. This is used to define the
                    // relative change of the initial length scale.
                    if (p_assembly)
                    {
                        mfem::Mult(grad_v_ref(q), Jinv, sgrad_v);
                    }
                    else
                    {
                        v.GetVectorGradient(*T, sgrad_v);
                    }
                    sgrad_v.Symmetrize();
                    double eig_val_data[3], eig_vec_data[9];
                    if (dim==1)
                    {
                        eig_val_data[0] = sgrad_v(0, 0);
                        eig_vec_data[0] = 1.;
                    }
                    else {
                        sgrad_v.CalcEigenvalues(eig_val_data, eig_vec_data);
                    }
                    Vector compr_dir(eig_vec_data, dim);
                    // Computes the initial->physical transformation Jacobian.
                    mfem::Mult(Jpr, quad_data.Jac0inv(z_id*nqp + q), Jpi);
                    Vector ph_dir(dim);
                    Jpi.Mult(compr_dir, ph_dir);
                    // Change of the initial mesh size in the compression direction.
                    const double h = quad_data.h0 * ph_dir.Norml2() /
                                     compr_dir.Norml2();

                    // Measure of maximal compression.
                    const double mu = eig_val_data[0];
                    visc_coeff = 2.0 * rho * h * h * fabs(mu);
                    // The following represents a "smooth" version of the statement
                    // "if (mu < 0) visc_coeff += 0.5 rho h sound_speed".  Note that
                    // eps must be scaled appropriately if a different unit system is
                    // being used.
                    const double eps = 1e-12;
                    visc_coeff += 0.5 * rho * h * sound_speed *
                                  (1.0 - smooth_step_01(mu - 2.0 * eps, eps));

                    stress.Add(visc_coeff, sgrad_v);
                }

                // Time step estimate at the point. Here the more relevant length
                // scale is related to the actual mesh deformation; we use the min
                // singular value of the ref->physical Jacobian. In addition, the
                // time step estimate should be aware of the presence of shocks.
                const double h_min =
                    Jpr.CalcSingularvalue(dim-1) / (double) H1FESpace.GetOrder(0);
                const double inv_dt = sound_speed / h_min +
                                      2.5 * visc_coeff / rho / h_min / h_min;
                if (min_detJ < 0.0)
                {
                    // This will force repetition of the step with smaller dt.
                    quad_data.dt_est = 0.0;
                }
                else
                {
                    quad_data.dt_est = min(quad_data.dt_est, cfl * (1.0 / inv_dt) );
                }

                // Quadrature data for partial assembly of the force operator.
                MultABt(stress, Jinv, stressJiT);
                stressJiT *= integ_rule.IntPoint(q).weight * detJ;
                for (int vd = 0 ; vd < dim; vd++)
                {
                    for (int gd = 0; gd < dim; gd++)
                    {
                        quad_data.stressJinvT(vd)(z_id*nqp + q, gd) =
                            stressJiT(vd, gd);
                    }
                }
            }
            ++z_id;
        }
    }

    delete [] gamma_b;
    delete [] rho_b;
    delete [] e_b;
    delete [] p_b;
    delete [] cs_b;
    delete [] Jpr_b;
    quad_data_is_current = true;
    forcemat_is_assembled = false;

    timer.sw_qdata.Stop();
    timer.quad_tstep += nzones;
}

void LagrangianHydroOperator::AssembleForceMatrix() const
{
    if (forcemat_is_assembled || p_assembly) {
        return;
    }

    Force = 0.0;
    timer.sw_force.Start();
    Force.Assemble();
    timer.sw_force.Stop();

    forcemat_is_assembled = true;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
