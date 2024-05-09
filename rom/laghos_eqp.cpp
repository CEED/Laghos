#include "laghos_rom.hpp"
#include "laghos_utils.hpp"

#include "linalg/NNLS.h"

void SolveNNLS(const int rank, const double nnls_tol, const int maxNNLSnnz,
               const bool useLQ, CAROM::Vector const& w, CAROM::Matrix & Gt,
               CAROM::Vector & sol)
{
    CAROM::NNLSSolver nnls(nnls_tol, 0, maxNNLSnnz, 2);

    // G.mult(w, rhs_ub);  // rhs = Gw
    // rhs = Gw. Note that by using Gt and multTranspose, we do parallel communication.

    CAROM::Vector rhs_Gw(Gt.numColumns(), false);
    Gt.transposeMult(w, rhs_Gw);

    if (useLQ)
    {

        // Compute Q^T of the LQ factorization of G.
        CAROM::Matrix* Qt_ptr;
        Qt_ptr = Gt.qr_factorize();

        CAROM::Matrix Qt(Qt_ptr->getData(), Qt_ptr->numRows(),
                         Qt_ptr->numColumns(), Qt_ptr->distributed(),
                         false);

        // Compute L of the factorization; L is lower triangular.
        // G = L * Q --> L = G * Q^T
        CAROM::Matrix L(Qt.numColumns(), Qt.numColumns(), false);
        Gt.transposeMult(Qt, L);

        // Check for nearly linearly dependent Q rows.
        // This is achieved by checking the magnitude of the diagonal
        // values of L.
        std::vector<int> row_ind;
        for (int i = 0; i < L.numRows(); ++i)
        {
            if (std::abs(L.item(i, i)) < 1e-12)
            {
                row_ind.push_back(i);
                std::cout << i << ", ";
            }
        }

        cout << "\nFound " << row_ind.size() << " / " << L.numRows()
             << " nearly linearly dependent constraints.\n";

        // Compute the RHS vector.
        CAROM::Vector rhs_ub(Qt.numColumns(), false);
        Qt.transposeMult(w, rhs_ub);

        // Compute the new RHS tolerance values.
        const double delta = 1.0e-11;
        CAROM::Vector delta_new(rhs_ub.dim(), false);
        for (int i = 0; i < delta_new.dim(); ++i)
        {
            double denominator = (i + 1) * std::abs(L.item(i, i));
            if (std::abs(denominator) < delta)
                delta_new(i) = 1.0;
            else
                delta_new(i) = delta / denominator;

            for (int j = i + 1; j < delta_new.dim(); ++j)
            {
                denominator = (j + 1) * std::abs(L.item(j, i));

                double temp;
                if (std::abs(denominator) < delta)
                    temp = 1.0;
                else
                    temp = delta / denominator;

                if (temp < delta_new(i))
                    delta_new(i) = temp;
            }
        }

        // Compute the upper and lower bound RHS vectors.
        CAROM::Vector rhs_lb(rhs_ub);
        for (int i = 0; i < rhs_ub.dim(); ++i)
        {
            rhs_lb(i) -= delta_new(i);
            rhs_ub(i) += delta_new(i);
        }

        // Call the NNLS solver.
        nnls.solve_parallel_with_scalapack(Qt, rhs_lb, rhs_ub, sol);

        delete Qt_ptr;
    }
    else
    {
        CAROM::Vector rhs_ub(rhs_Gw);
        CAROM::Vector rhs_lb(rhs_Gw);

        const double delta = 1.0e-11;
        for (int i=0; i<rhs_ub.dim(); ++i)
        {
            rhs_lb(i) -= delta;
            rhs_ub(i) += delta;
        }

        //nnls.normalize_constraints(Gt, rhs_lb, rhs_ub);
        nnls.solve_parallel_with_scalapack(Gt, rhs_lb, rhs_ub, sol);
    }

    int nnz = 0;
    for (int i=0; i<sol.dim(); ++i)
    {
        if (sol(i) != 0.0)
        {
            nnz++;
        }
    }

    cout << rank << ": Number of nonzeros in NNLS solution: " << nnz
         << ", out of " << sol.dim() << endl;

    MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Global number of nonzeros in NNLS solution: " << nnz << endl;

    // Check residual of NNLS solution
    CAROM::Vector res(Gt.numColumns(), false);
    Gt.transposeMult(sol, res);

    const double normGsol = res.norm();
    const double normRHS = rhs_Gw.norm();

    res -= rhs_Gw;
    const double relNorm = res.norm() / std::max(normGsol, normRHS);
    cout << rank << ": relative residual norm for NNLS solution of Gs = Gw: " <<
         relNorm << endl;
}

void WriteSolutionNNLS(std::vector<int> const& indices, std::vector<double> const& sol,
                       const string filename)
{
    std::ofstream outfile(filename);

    const int n = indices.size();
    MFEM_VERIFY(n == sol.size(), "");

    for (int i=0; i<n; ++i)
    {
        outfile << indices[i] << " " << sol[i] << "\n";
    }

    outfile.close();
}

void ExtractNonzeros(CAROM::Vector const& v, std::vector<int> & indices,
                     std::vector<double> & nz)
{
    nz.clear();
    indices.clear();
    for (int i=0; i<v.dim(); ++i)
    {
        if (v(i) != 0.0)
        {
            indices.push_back(i);
            nz.push_back(v(i));
        }
    }
}

void ExtractMatrixElementRowsAndWrite(std::set<int> const& elems,
                                      const ParFiniteElementSpace *fespace,
                                      CAROM::Matrix const& Wmat,
                                      std::string filename, int rank, int nprocs)
{
    bool negdof = false;
    const int ncol = Wmat.numColumns();

    std::vector<int> elemDofs;
    Array<int> vdofs;
    int ndof = 0;
    for (auto e : elems)
    {
        fespace->GetElementVDofs(e, vdofs);
        if (ndof == 0)
        {
            ndof = vdofs.Size();
        }
        else
        {
            MFEM_VERIFY(ndof == vdofs.Size(), "");
        }

        for (auto dof : vdofs)
        {
            elemDofs.push_back(dof < 0 ? -1-dof : dof);
            if (dof < 0) negdof = true;
        }
    }

    MFEM_VERIFY(ndof * elems.size() == elemDofs.size(), "");
    MFEM_VERIFY(!negdof, "negdof"); // If negative, flip sign of DOF value.

    CAROM::Matrix W_elems;

    if (elemDofs.size() > 0)
    {
        W_elems.setSize(elemDofs.size(), ncol);

        for (int j=0; j<ncol; ++j)
        {
            for (int i=0; i<elemDofs.size(); ++i)
            {
                W_elems(i,j) = Wmat(elemDofs[i],j);
            }
        }
    }

    // Gather all entries of W_elems to root.
    const int numEntries = elemDofs.size() > 0 ?
                           W_elems.numRows() * W_elems.numColumns() : 0;

    vector<double> allData;
    Double_Gatherv(numEntries, numEntries > 0 ? W_elems.getData() : nullptr,
                   0, rank, nprocs, MPI_COMM_WORLD, allData);

    if (rank == 0)
    {
        CAROM::Matrix allW_elems(allData.data(), allData.size() / ncol, ncol, false, false);
        allW_elems.write(filename);
    }
}

// This function is based on ForceIntegrator::AssembleElementMatrix2.
void ComputeElementRowOfG_V(const IntegrationRule *ir,
                            hydrodynamics::QuadratureData const& quad_data,
                            Vector const& v_e,
                            FiniteElement const& test_fe, FiniteElement const& trial_fe,
                            const int zone_id, Vector & r)
{
    MFEM_VERIFY(r.Size() == ir->GetNPoints(), "");

    const int nqp = ir->GetNPoints();
    const int dim = trial_fe.GetDim();
    const int h1dofs_cnt = test_fe.GetDof();
    const int l2dofs_cnt = trial_fe.GetDof();

    MFEM_VERIFY(v_e.Size() == h1dofs_cnt*dim, "");

    DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
    Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt * dim);

    Vector unitE(l2dofs_cnt);
    unitE = 1.0;

    Vector rhs(h1dofs_cnt*dim);

    for (int q = 0; q < nqp; q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);

        // Form stress:grad_shape at the current point.
        test_fe.CalcDShape(ip, vshape);
        for (int i = 0; i < h1dofs_cnt; i++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(i, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(i, vd) +=
                        quad_data.stressJinvT(vd)(zone_id*nqp + q, gd) * vshape(i,gd);
                }
            }
        }

        // NOTE: UpdateQuadratureData includes ip.weight as a factor in quad_data.stressJinvT,
        // set by LagrangianHydroOperator::UpdateQuadratureData.
        loc_force *= 1.0 / ip.weight;  // Divide by exact quadrature weight

        trial_fe.CalcShape(ip, shape);

        // Compute the inner product, on this element, with the j-th V basis vector.
        r[q] = (v_e * Vloc_force) * (shape * unitE);
    }
}

// This function is based on ForceIntegrator::AssembleElementMatrix2.
void ComputeElementRowOfG_E(const IntegrationRule *ir,
                            hydrodynamics::QuadratureData const& quad_data,
                            Vector const& w_e, Vector const& v_e,
                            FiniteElement const& test_fe, FiniteElement const& trial_fe,
                            const int zone_id, Vector & r)
{
    MFEM_VERIFY(r.Size() == ir->GetNPoints(), "");

    const int nqp = ir->GetNPoints();
    const int dim = trial_fe.GetDim();
    const int h1dofs_cnt = test_fe.GetDof();
    const int l2dofs_cnt = trial_fe.GetDof();

    MFEM_VERIFY(v_e.Size() == h1dofs_cnt*dim, "");
    MFEM_VERIFY(w_e.Size() == l2dofs_cnt, "");

    DenseMatrix vshape(h1dofs_cnt, dim), loc_force(h1dofs_cnt, dim);
    Vector shape(l2dofs_cnt), Vloc_force(loc_force.Data(), h1dofs_cnt * dim);

    Vector rhs(h1dofs_cnt*dim);

    for (int q = 0; q < nqp; q++)
    {
        const IntegrationPoint &ip = ir->IntPoint(q);

        // Form stress:grad_shape at the current point.
        test_fe.CalcDShape(ip, vshape);
        for (int i = 0; i < h1dofs_cnt; i++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(i, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(i, vd) +=
                        quad_data.stressJinvT(vd)(zone_id*nqp + q, gd) * vshape(i,gd);
                }
            }
        }

        // NOTE: UpdateQuadratureData includes ip.weight as a factor in quad_data.stressJinvT,
        // set by LagrangianHydroOperator::UpdateQuadratureData.
        loc_force *= 1.0 / ip.weight;  // Divide by exact quadrature weight

        trial_fe.CalcShape(ip, shape);

        // Compute the inner product, on this element, with the j-th V basis vector.
        r[q] = (v_e * Vloc_force) * (shape * w_e);
    }
}

void ROM_Basis::SetupEQP_Force(std::vector<const CAROM::Matrix*> snapX,
                               std::vector<const CAROM::Matrix*> snapV,
                               std::vector<const CAROM::Matrix*> snapE,
                               const CAROM::Matrix* basisV,
                               const CAROM::Matrix* basisE,
                               ROM_Options const& input, std::set<int> & elems)
{
    MFEM_VERIFY(basisV->numRows() == input.H1FESpace->GetTrueVSize(), "");
    MFEM_VERIFY(basisE->numRows() == input.L2FESpace->GetTrueVSize(), "");

    for (auto snap : snapX)
    {
        MFEM_VERIFY(snap->numRows() == input.H1FESpace->GetTrueVSize(), "");
    }

    for (auto snap : snapV)
    {
        MFEM_VERIFY(snap->numRows() == input.H1FESpace->GetTrueVSize(), "");
    }

    for (auto snap : snapE)
    {
        MFEM_VERIFY(snap->numRows() == input.L2FESpace->GetTrueVSize(), "");
    }

    cout << "WINDOW " << input.window << endl;
    SetupEQP_Force_Eq(snapX, snapV, snapE, basisV, basisE, input, false, elems);
    SetupEQP_Force_Eq(snapX, snapV, snapE, basisV, basisE, input, true, elems);

    //WriteSampleMeshEQP(input, elems);

    // For the parallel case, gather local elems and construct global elems.
    std::vector<int> globalElems;
    {
        const int localNumElems = elems.size();

        std::vector<int> counts(nprocs);
        std::vector<int> offsets(nprocs);

        MPI_Gather(&localNumElems, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        offsets[0] = 0;
        for (int i=1; i<nprocs; ++i)
            offsets[i] = offsets[i-1] + counts[i-1];

        std::vector<int> localElems(elems.begin(), elems.end());
        Int_Gatherv(localNumElems, localNumElems > 0 ? localElems.data() : nullptr,
                    0, rank, nprocs, MPI_COMM_WORLD, globalElems);

        const int ne = input.H1FESpace->GetNE();
        std::vector<int> allne(nprocs);
        MPI_Gather(&ne, 1, MPI_INT, allne.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            int os = 0;
            for (int i=1; i<nprocs; ++i)
            {
                os += allne[i-1];
                for (int j=offsets[i]; j<offsets[i] + counts[i]; ++j)
                {
                    globalElems[j] += os;
                }
            }
        }
    }

    // Write elems to file
    if (rank == 0)
    {
        std::ofstream outfile("run/nnlsElems" + std::to_string(input.window));

        for (auto e : globalElems)
            outfile << e << endl;
        outfile.close();
    }

    // Construct Wmat and write out rows corresponding to elements in elems to file.
    const int tsize_H1 = input.H1FESpace->GetTrueVSize();
    const int tsize_L2 = input.L2FESpace->GetTrueVSize();

    CAROM::Matrix Wmat(H1size, rdimv, true);
    CAROM::Matrix Wmat_E(L2size, rdime, true);

    // Compute Wmat
    Vector vj(tsize_H1);
    Vector Mvj(H1size);
    for (int j=0; j<rdimv; ++j)
    {
        GetBasisVectorV(false, j, vj);
        gfH1->SetFromTrueDofs(vj);

#ifdef EQP_NO_Minv
        for (int i=0; i<H1size; ++i)
            Wmat(i,j) = (*gfH1)[i];
#else
        input.FOMoper->MultMvInv(*gfH1, Mvj);

        for (int i=0; i<H1size; ++i)
            Wmat(i,j) = Mvj[i];
#endif
    }

    // Compute Wmat_E
    vj.SetSize(tsize_L2);
    Mvj.SetSize(L2size);
    for (int j=0; j<rdime; ++j)
    {
        GetBasisVectorE(false, j, vj);
        gfL2->SetFromTrueDofs(vj);
#ifdef EQP_NO_Minv
        for (int i=0; i<L2size; ++i)
            Wmat_E(i,j) = (*gfL2)[i];
#else
        input.FOMoper->MultMeInv(*gfL2, Mvj);

        for (int i=0; i<L2size; ++i)
            Wmat_E(i,j) = Mvj[i];
#endif
    }

    ExtractMatrixElementRowsAndWrite(elems, input.H1FESpace, Wmat,
                                     *input.basename + "/WelemsV" +
                                     std::to_string(input.window),
                                     input.rank, input.nprocs);

    ExtractMatrixElementRowsAndWrite(elems, input.L2FESpace, Wmat_E,
                                     *input.basename + "/WelemsE" +
                                     std::to_string(input.window),
                                     input.rank, input.nprocs);
}

void ROM_Basis::SetupEQP_Force_Eq(std::vector<const CAROM::Matrix*> snapX,
                                  std::vector<const CAROM::Matrix*> snapV,
                                  std::vector<const CAROM::Matrix*> snapE,
                                  const CAROM::Matrix* basisV,
                                  const CAROM::Matrix* basisE,
                                  ROM_Options const& input,
                                  bool equationE,
                                  std::set<int> & elems)
{
    const IntegrationRule *ir0 = input.FOMoper->GetIntegrationRule();
    const int nqe = ir0->GetNPoints();
    const int ne = input.H1FESpace->GetNE();
    const int NQ = ne * nqe;
    const int NB = equationE ? basisE->numColumns() : basisV->numColumns();

    const int nsets = input.numOfflineParameters;
    MFEM_VERIFY(nsets == snapX.size() && nsets == snapV.size() &&
                nsets == snapE.size() && nsets > 0, "");

    Array<int> numSnapVar(3);
    std::vector<Array<int>> allNumSnapVar(3);
    for (int i=0; i<3; ++i)
        allNumSnapVar[i].SetSize(nsets);

    Array<int> allnsnap(nsets);

    for (int i=0; i<nsets; ++i)
    {
        allNumSnapVar[0][i] = snapX[i]->numColumns();
        allNumSnapVar[1][i] = snapV[i]->numColumns();
        allNumSnapVar[2][i] = snapE[i]->numColumns();

        // TODO: is there a better way?
        allnsnap[i] = allNumSnapVar[0][i];
        for (int j=1; j<3; ++j)
        {
            allnsnap[i] = std::max(allnsnap[i], allNumSnapVar[j][i]);
        }
    }

    for (int i=0; i<3; ++i)
        numSnapVar[i] = allNumSnapVar[i].Sum();

    const int nsnap = numSnapVar.Max();

    std::vector<Array<int>> numSkipped(3);
    for (int i=0; i<3; ++i)
    {
        numSkipped[i].SetSize(nsets);

        for (int j=0; j<nsets; ++j)
        {
            numSkipped[i][j] = allnsnap[j] - allNumSnapVar[i][j];
        }

        MFEM_VERIFY(numSkipped[i].Max() <= 1, "");
    }

    Vector r(nqe);

    // Compute G of size (NB * (nsnap+1)) x NQ, storing its transpose Gt.
    CAROM::Matrix Gt(NQ, NB * (nsnap+1), true);
    cout << "NNLS using " << NB << " basis dim for equation E " << equationE
         << " and " << nsnap << " snapshots" << endl;

    // For 0 <= j < NB, 0 <= i <= nsnap, 0 <= e < ne, 0 <= m < nqe,
    // G(j + (i*NB), (e*nqe) + m)
    // is the coefficient of e_j^T M_e^{-1} F(v_i,e_i,x_i)^T v_i at point m of
    // element e, with respect to the integration rule weight at that point,
    // where the "exact" quadrature solution is ir0->GetWeights().

    Vector v_i(tH1size);
    Vector x_i(tH1size);
    Vector e_i(tL2size);
    Vector g_i(tL2size);

    Vector w_j_e, v_i_e, v_j_e;

    Vector S((2*input.H1FESpace->GetVSize()) + input.L2FESpace->GetVSize());
    Vector S_v(S, input.H1FESpace->GetVSize(), input.H1FESpace->GetVSize());  // Subvector

    MFEM_VERIFY(tH1size == basisV->numRows(), "");
    MFEM_VERIFY(tL2size == basisE->numRows(), "");
    CAROM::Matrix W(equationE ? L2size : H1size, NB, true);

    ParGridFunction gf2H1(*gfH1);

    for (int j=0; j<NB; ++j)
    {
        if (equationE)
        {
            for (int i=0; i<tL2size; ++i)
                e_i[i] = (*basisE)(i,j);

#ifdef EQP_NO_Minv
            for (int i=0; i<tL2size; ++i)
                W(i,j) = e_i[i];  // TODO: just set to basisE?
#else
            input.FOMoper->MultMeInv(e_i, g_i);

            for (int i=0; i<tL2size; ++i)
                W(i,j) = g_i[i];
#endif
        }
        else
        {
            for (int i=0; i<tH1size; ++i)
                v_i[i] = (*basisV)(i,j);

            gfH1->SetFromTrueDofs(v_i);
#ifdef EQP_NO_Minv
            for (int i=0; i<H1size; ++i)
                W(i,j) = (*gfH1)[i];
#else
            input.FOMoper->MultMvInv(*gfH1, gf2H1);

            for (int i=0; i<H1size; ++i)
                W(i,j) = gf2H1[i];
#endif
        }
    }

    Array<double> const& w_el = ir0->GetWeights();
    MFEM_VERIFY(w_el.Size() == nqe, "");

    int oss = 0;
    for (int s=-1; s<nsets; ++s)
    {
        const int nsnap_s = (s == -1) ? 1 : allnsnap[s];
        for (int i=0; i<nsnap_s; ++i)
        {
            if (s == -1)  // Use the initial state as the first snapshot.
            {
                v_i = 0.0;
                x_i = 0.0;
                e_i = 0.0;
            }
            else
            {
                if (i == 0 && numSkipped[0][s] == 1)
                {
                    x_i = 0.0;
                }
                else
                {
                    for (int j = 0; j < tH1size; ++j)
                        x_i[j] = (*snapX[s])(j, i - numSkipped[0][s]);
                }

                if (i == 0 && numSkipped[1][s] == 1)
                    v_i = 0.0;
                else
                {
                    for (int j = 0; j < tH1size; ++j)
                        v_i[j] = (*snapV[s])(j, i - numSkipped[1][s]);
                }

                if (i == 0 && numSkipped[2][s] == 1)
                    e_i = 0.0;
                else
                {
                    for (int j = 0; j < tL2size; ++j)
                    {
                        e_i[j] = (*snapE[s])(j, i - numSkipped[2][s]);
                    }
                }
            }

            SetStateFromTrueDOFs(x_i, v_i, e_i, S);

            // NOTE: after SetStateFromTrueDOFs, gfH1 is the V-component of S
            input.FOMoper->ResetQuadratureData();
            input.FOMoper->GetTimeStepEstimate(S);  // Call UpdateQuadratureData
            input.FOMoper->ResetQuadratureData();

            for (int j=0; j<NB; ++j)
            {
                if (equationE)
                {
                    for (int k = 0; k < basisE->numRows(); ++k)
                    {
                        mfL2[k] = W(k, j);
                    }

                    gfL2->SetFromTrueDofs(mfL2);
                    *gfH1 = S_v;
                }
                else
                {
                    for (int k = 0; k < H1size; ++k) (*gfH1)[k] = W(k, j);
                }

                for (int e=0; e<ne; ++e)
                {
                    if (equationE)
                    {
                        gfL2->GetElementDofValues(e, w_j_e);
                        gfH1->GetElementDofValues(e, v_i_e);

                        ComputeElementRowOfG_E(ir0, input.FOMoper->GetQuadData(),
                                               w_j_e, v_i_e,
                                               *input.H1FESpace->GetFE(e),
                                               *input.L2FESpace->GetFE(e), e, r);
                    }
                    else
                    {
                        gfH1->GetElementDofValues(e, v_j_e);
                        ComputeElementRowOfG_V(ir0, input.FOMoper->GetQuadData(),
                                               v_j_e, *input.H1FESpace->GetFE(e),
                                               *input.L2FESpace->GetFE(e), e, r);
                    }

                    for (int m=0; m<nqe; ++m)
                    {
                        Gt((e*nqe) + m, j + ((oss + i)*NB)) = r[m];
                    }
                }  // e
            }  // j
        }  // i

        oss += nsnap_s;
    }  // s

    // Rescale every Gt column (NNLS equation) by its max absolute value.
    // It seems to help the NNLS solver significantly.
    Gt.rescale_cols_max();

    CAROM::Vector w(ne * nqe, true);
    for (int i=0; i<ne; ++i)
    {
        for (int j=0; j<nqe; ++j)
            w((i*nqe) + j) = w_el[j];
    }

    CAROM::Vector sol(ne * nqe, true);
    SolveNNLS(input.rank, input.tolNNLS, input.maxNNLSnnz, input.LQ_NNLS,
              w, Gt, sol);

    std::vector<double> solnz; // Solution nonzeros
    std::vector<int> indices;
    ExtractNonzeros(sol, indices, solnz);

    int prev = -1;
    for (auto i : indices)
    {
        const int elem = i / nqe;
        if (elem != prev)
        {
            elems.insert(elem);
            prev = elem;
        }
    }

    // For the parallel case, convert local indices to global on root.
    std::vector<int> allne(nprocs);
    MPI_Gather(&ne, 1, MPI_INT, allne.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> globalIndices;
    std::vector<double> globalSol;

    const int localNumIndices = indices.size();

    Int_Gatherv(localNumIndices, indices.data(), 0, rank, nprocs, MPI_COMM_WORLD, globalIndices);
    Double_Gatherv(localNumIndices, solnz.data(), 0, rank, nprocs, MPI_COMM_WORLD, globalSol);

    std::vector<int> counts(nprocs);
    std::vector<int> offsets(nprocs);

    MPI_Gather(&localNumIndices, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    offsets[0] = 0;
    for (int i=1; i<nprocs; ++i)
        offsets[i] = offsets[i-1] + counts[i-1];

    if (rank == 0)
    {
        int os = 0;
        for (int i=1; i<nprocs; ++i)
        {
            os += allne[i-1] * nqe;
            for (int j=offsets[i]; j<offsets[i] + counts[i]; ++j)
            {
                globalIndices[j] += os;
            }
        }

        const std::string varName = equationE ? "E" : "V";
        WriteSolutionNNLS(globalIndices, globalSol, "run/nnls" + varName +
                          std::to_string(input.window));
    }
}

void ROM_Operator::InitEQP() const
{
    operSP->SetPointsEQP(eqpI);
    operSP->SetPointsEQP(eqpI_E);
}

void ROM_Operator::ForceIntegratorEQP(Vector & res) const
{
    MFEM_ABORT("REMOVE");
    const IntegrationRule *ir = operFOM->GetIntegrationRule();
    const int rdim = basis->GetDimV();
    MFEM_VERIFY(eqpI.size() == eqpW.size(), "");
    MFEM_VERIFY(res.Size() == rdim, "");

    const int nqe = ir->GetWeights().Size();

    DenseMatrix vshape, loc_force;
    Vector shape, unitE, rhs;

    Array<int> vdofs;

    Vector v_e;

    res = 0.0;

    int eprev = -1;
    int dof = 0;
    int spaceDim = 0;

    const hydrodynamics::QuadratureData & quad_data = operFOM->GetQuadData();

    // TODO: optimize by storing some intermediate computations.

    const FiniteElement *test_fe = nullptr;
    const FiniteElement *trial_fe = nullptr;

    if (!eqp_init)
    {
        eqp_init = true;

        std::vector<int> elements;
        for (int i=0; i<eqpW.size(); ++i)
        {
            const int e = eqpI[i] / nqe;  // Element index
            if (e != eprev)
            {
                elements.push_back(e);
                eprev = e;
            }
        }
        eprev = -1;

        // TODO: eliminate the following code, since W_elems is just read from file?
        bool negdof = false;

        std::vector<int> elemDofs;
        for (auto e : elements)
        {
            H1spaceFOM->GetElementVDofs(e, vdofs);
            if (nvdof == 0)
            {
                nvdof = vdofs.Size();
            }
            else
            {
                MFEM_VERIFY(nvdof == vdofs.Size(), "");
            }

            for (auto dof : vdofs)
            {
                elemDofs.push_back(dof < 0 ? -1-dof : dof);
                if (dof < 0) negdof = true;
            }
        }

        MFEM_VERIFY(nvdof * elements.size() == elemDofs.size(), "");
        MFEM_VERIFY(!negdof, "negdof"); // If negative, flip sign of DOF value.

        // TODO: get the basename "run" and window from ROM_Options?
        //W_elems.read(*input.basename + "/WelemsV" + std::to_string(input.window));
        W_elems.read("run/WelemsV" + std::to_string(window));
        MFEM_VERIFY(W_elems.numRows() == elemDofs.size() && W_elems.numColumns() == rdim, "");
    }

    int elemIndex = -1;

    for (int i=0; i<eqpW.size(); ++i)
    {
        const int e = eqpI[i] / nqe;  // Element index
        // Local (element) index of the quadrature point
        const int qpi = eqpI[i] - (e*nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev)  // Update element transformation
        {
            elemIndex++;

            test_fe = H1spaceFOM->GetFE(e);
            trial_fe = L2spaceFOM->GetFE(e);

            MFEM_VERIFY(nvdof == test_fe->GetDim() * test_fe->GetDof(), ""); // TODO: remove this sanity check

            eprev = e;
        }

        // Integrate at the current point

        // NOTE: quad_data is updated at the FOM level by LagrangianHydroOperator::UpdateQuadratureData.
        // NOTE: quad_data includes ip.weight as a factor in quad_data.stressJinvT, so we divide it out here.
        // TODO: reduce this UpdateQuadratureData function to the EQP points.

        const int h1dofs_cnt = test_fe->GetDof();
        const int dim = trial_fe->GetDim();

        const int l2dofs_cnt = trial_fe->GetDof();

        vshape.SetSize(h1dofs_cnt, dim);
        loc_force.SetSize(h1dofs_cnt, dim);

        shape.SetSize(l2dofs_cnt);

        // Form stress:grad_shape at the current point.
        test_fe->CalcDShape(ip, vshape);
        for (int k = 0; k < h1dofs_cnt; k++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(k, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(k, vd) +=
                        quad_data.stressJinvT(vd)(e*nqe + qpi, gd) * vshape(k,gd);
                }
            }
        }

        loc_force *= eqpW[i] / ip.weight;  // Replace exact quadrature weight with EQP weight.

        trial_fe->CalcShape(ip, shape);

        Vector Vloc_force(loc_force.Data(), loc_force.NumRows() * loc_force.NumCols());

        unitE.SetSize(shape.Size());
        unitE = 1.0;

        rhs.SetSize(h1dofs_cnt * dim);

        v_e.SetSize(rhs.Size());

        const int eos = elemIndex * nvdof;
        for (int j=0; j<rdim; ++j)
        {
            for (int k=0; k<nvdof; ++k)
                v_e[k] = W_elems(eos + k, j);

            // Compute the inner product, on this element, with the j-th W vector.
            res[j] += (v_e * Vloc_force) * (shape * unitE);
        }  // Loop (j) over V basis vectors
    } // Loop (i) over EQP points

    MPI_Allreduce(MPI_IN_PLACE, res.GetData(), res.Size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
}

void ROM_Operator::ForceIntegratorEQP_SP() const
{
    const IntegrationRule *ir = operSP->GetIntegrationRule();
    const int rdim = basis->GetDimV();
    MFEM_VERIFY(eqpI.size() == eqpW.size(), "");
    eqpFv.SetSize(rdim);

    const int nqe = ir->GetWeights().Size();

    DenseMatrix vshape, loc_force;
    Vector shape;

    Array<int> vdofs;

    eqpFv = 0.0;

    int eprev = -1;
    int dof = 0;
    int spaceDim = 0;

    const hydrodynamics::QuadratureData & quad_data = operSP->GetQuadData();

    // TODO: optimize by storing some intermediate computations.

    const FiniteElement *test_fe = nullptr;
    const FiniteElement *trial_fe = nullptr;

    int elemIndex = -1;

    for (int i=0; i<eqpW.size(); ++i)
    {
        const int e = eqpI[i] / nqe;  // Element index
        // Local (element) index of the quadrature point
        const int qpi = eqpI[i] - (e*nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev)  // Update element transformation
        {
            elemIndex++;

            test_fe = H1FESpaceSP->GetFE(e);
            trial_fe = L2FESpaceSP->GetFE(e);

            MFEM_VERIFY(nvdof == test_fe->GetDim() * test_fe->GetDof(), ""); // TODO: remove this sanity check

            eprev = e;
        }

        // Integrate at the current point

        // NOTE: quad_data is updated at the FOM level by LagrangianHydroOperator::UpdateQuadratureData.
        // NOTE: quad_data includes ip.weight as a factor in quad_data.stressJinvT, so we divide it out here.
        // TODO: reduce this UpdateQuadratureData function to the EQP points.

        const int h1dofs_cnt = test_fe->GetDof();
        const int dim = trial_fe->GetDim();

        const int l2dofs_cnt = trial_fe->GetDof();

        vshape.SetSize(h1dofs_cnt, dim);
        loc_force.SetSize(h1dofs_cnt, dim);
        double *force_data = loc_force.Data();

        shape.SetSize(l2dofs_cnt);

        // Form stress:grad_shape at the current point.
        test_fe->CalcDShape(ip, vshape);
        for (int k = 0; k < h1dofs_cnt; k++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(k, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(k, vd) +=
                        quad_data.stressJinvT(vd)(e*nqe + qpi, gd) * vshape(k, gd);
                }
            }
        }

        loc_force *= eqpW[i] / ip.weight;  // Replace exact quadrature weight with EQP weight.

        trial_fe->CalcShape(ip, shape);
        const double ip_e = shape.Sum();

        const int eos = e * nvdof;
        for (int j=0; j<rdim; ++j)
        {
            double ip_f = 0.0;
            for (int k=0; k<nvdof; ++k)
            {
                ip_f += W_elems(eos + k, j) * force_data[k];
            }

            // Compute the inner product, on this element, with the j-th W vector.
            eqpFv[j] += ip_f * ip_e;
        }  // Loop (j) over V basis vectors
    } // Loop (i) over EQP points

    MPI_Allreduce(MPI_IN_PLACE, eqpFv.GetData(), eqpFv.Size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    eqpFv.Neg();  // Matching LagrangianHydroOperator::SolveVelocity

#ifdef EQP_NO_Minv
    Vector invMvF(rdim);
    invMvROM.Mult(eqpFv, invMvF);
    eqpFv = invMvF;
#endif
}

void ROM_Operator::ForceIntegratorEQP_E(Vector const& v, Vector & res) const
{
    MFEM_ABORT("REMOVE");

    const IntegrationRule *ir = operFOM->GetIntegrationRule();
    const int rdim = basis->GetDimE();
    MFEM_VERIFY(eqpI_E.size() == eqpW_E.size(), "");
    MFEM_VERIFY(res.Size() == rdim, "");

    const int nqe = ir->GetWeights().Size();

    DenseMatrix vshape, loc_force;
    Vector shape, rhs;

    Array<int> vdofs, edofs;

    Vector v_e, w_e;

    res = 0.0;

    int eprev = -1;
    int dof = 0;
    int spaceDim = 0;

    const hydrodynamics::QuadratureData & quad_data = operFOM->GetQuadData();

    // TODO: optimize by storing some intermediate computations.

    const FiniteElement *test_fe = nullptr;
    const FiniteElement *trial_fe = nullptr;

    if (!eqp_init_E)
    {
        eqp_init_E = true;

        std::vector<int> elements;
        for (int i=0; i<eqpW_E.size(); ++i)
        {
            const int e = eqpI_E[i] / nqe;  // Element index
            if (e != eprev)
            {
                elements.push_back(e);
                eprev = e;
            }
        }
        eprev = -1;

        bool negdof = false;

        std::vector<int> elemDofs;
        for (auto e : elements)
        {
            H1spaceFOM->GetElementVDofs(e, vdofs);
            if (nvdof == 0)
            {
                nvdof = vdofs.Size();
            }
            else
            {
                MFEM_VERIFY(nvdof == vdofs.Size(), "");
            }

            L2spaceFOM->GetElementVDofs(e, edofs);
            if (nedof == 0)
            {
                nedof = edofs.Size();
            }
            else
            {
                MFEM_VERIFY(nedof == edofs.Size(), "");
            }

            for (auto dof : edofs)
            {
                elemDofs.push_back(dof < 0 ? -1-dof : dof);
                if (dof < 0) negdof = true;
            }
        }

        MFEM_VERIFY(nedof * elements.size() == elemDofs.size(), "");
        MFEM_VERIFY(!negdof, "negdof"); // If negative, flip sign of DOF value.

        // TODO: get the basename "run" from ROM_Options?
        //W_E_elems.read(*input.basename + "/WelemsE" + std::to_string(input.window));
        W_E_elems.read("run/WelemsE" + std::to_string(window));
        MFEM_VERIFY(W_E_elems.numRows() == elemDofs.size() && W_E_elems.numColumns() == rdim, "");
    }

    int elemIndex = -1;

    for (int i=0; i<eqpW_E.size(); ++i)
    {
        const int e = eqpI_E[i] / nqe;  // Element index
        // Local (element) index of the quadrature point
        const int qpi = eqpI_E[i] - (e*nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev)  // Update element transformation
        {
            elemIndex++;

            test_fe = H1spaceFOM->GetFE(e);
            trial_fe = L2spaceFOM->GetFE(e);

            MFEM_VERIFY(nvdof == test_fe->GetDim() * test_fe->GetDof(), ""); // TODO: remove this sanity check

            H1spaceFOM->GetElementVDofs(e, vdofs);
            MFEM_VERIFY(nvdof == vdofs.Size(), ""); // TODO: remove this sanity check

            v.GetSubVector(vdofs, v_e);

            eprev = e;
        }

        // Integrate at the current point

        // NOTE: quad_data is updated at the FOM level by LagrangianHydroOperator::UpdateQuadratureData.
        // NOTE: quad_data includes ip.weight as a factor in quad_data.stressJinvT, so we divide it out here.
        // TODO: reduce this UpdateQuadratureData function to the EQP points.

        const int h1dofs_cnt = test_fe->GetDof();
        const int dim = trial_fe->GetDim();

        const int l2dofs_cnt = trial_fe->GetDof();

        vshape.SetSize(h1dofs_cnt, dim);
        loc_force.SetSize(h1dofs_cnt, dim);

        shape.SetSize(l2dofs_cnt);

        // Form stress:grad_shape at the current point.
        test_fe->CalcDShape(ip, vshape);
        for (int k = 0; k < h1dofs_cnt; k++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(k, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(k, vd) +=
                        quad_data.stressJinvT(vd)(e*nqe + qpi, gd) * vshape(k,gd);
                }
            }
        }

        loc_force *= eqpW_E[i] / ip.weight;  // Replace exact quadrature weight with EQP weight.

        trial_fe->CalcShape(ip, shape);

        Vector Vloc_force(loc_force.Data(), loc_force.NumRows() * loc_force.NumCols());

        rhs.SetSize(h1dofs_cnt * dim);

        w_e.SetSize(nedof);

        const int eos = elemIndex * nedof;
        for (int j=0; j<rdim; ++j)
        {
            for (int k=0; k<nedof; ++k)
                w_e[k] = W_E_elems(eos + k, j);

            // Compute the inner product, on this element, with the j-th W vector.
            res[j] += (v_e * Vloc_force) * (shape * w_e);
        }  // Loop (j) over V basis vectors
    } // Loop (i) over EQP points

    MPI_Allreduce(MPI_IN_PLACE, res.GetData(), res.Size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
}

void ROM_Operator::ForceIntegratorEQP_E_SP(Vector const& v) const
{
    const IntegrationRule *ir = operSP->GetIntegrationRule();
    const int rdim = basis->GetDimE();
    MFEM_VERIFY(eqpI_E.size() == eqpW_E.size(), "");
    eqpFe.SetSize(rdim);

    const int nqe = ir->GetWeights().Size();

    DenseMatrix vshape, loc_force;
    Vector shape, rhs;

    Array<int> vdofs, edofs;

    Vector v_e, w_e;

    eqpFe = 0.0;

    int eprev = -1;
    int dof = 0;
    int spaceDim = 0;

    const hydrodynamics::QuadratureData & quad_data = operSP->GetQuadData();

    // TODO: optimize by storing some intermediate computations.

    const FiniteElement *test_fe = nullptr;
    const FiniteElement *trial_fe = nullptr;

    int elemIndex = -1;

    for (int i=0; i<eqpW_E.size(); ++i)
    {
        const int e = eqpI_E[i] / nqe;  // Element index
        // Local (element) index of the quadrature point
        const int qpi = eqpI_E[i] - (e*nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev)  // Update element transformation
        {
            elemIndex++;

            test_fe = H1FESpaceSP->GetFE(e);
            trial_fe = L2FESpaceSP->GetFE(e);

            MFEM_VERIFY(nvdof == test_fe->GetDim() * test_fe->GetDof(), ""); // TODO: remove this sanity check

            H1FESpaceSP->GetElementVDofs(e, vdofs);
            MFEM_VERIFY(nvdof == vdofs.Size(), ""); // TODO: remove this sanity check

            v.GetSubVector(vdofs, v_e);

            eprev = e;
        }

        // Integrate at the current point

        // NOTE: quad_data is updated at the FOM level by LagrangianHydroOperator::UpdateQuadratureData.
        // NOTE: quad_data includes ip.weight as a factor in quad_data.stressJinvT, so we divide it out here.
        // TODO: reduce this UpdateQuadratureData function to the EQP points.

        const int h1dofs_cnt = test_fe->GetDof();
        const int dim = trial_fe->GetDim();

        const int l2dofs_cnt = trial_fe->GetDof();

        vshape.SetSize(h1dofs_cnt, dim);
        loc_force.SetSize(h1dofs_cnt, dim);

        shape.SetSize(l2dofs_cnt);

        // Form stress:grad_shape at the current point.
        test_fe->CalcDShape(ip, vshape);
        for (int k = 0; k < h1dofs_cnt; k++)
        {
            for (int vd = 0; vd < dim; vd++) // Velocity components.
            {
                loc_force(k, vd) = 0.0;
                for (int gd = 0; gd < dim; gd++) // Gradient components.
                {
                    loc_force(k, vd) +=
                        quad_data.stressJinvT(vd)(e*nqe + qpi, gd) * vshape(k,gd);
                }
            }
        }

        loc_force *= eqpW_E[i] / ip.weight;  // Replace exact quadrature weight with EQP weight.

        trial_fe->CalcShape(ip, shape);

        Vector Vloc_force(loc_force.Data(), loc_force.NumRows() * loc_force.NumCols());

        rhs.SetSize(h1dofs_cnt * dim);

        w_e.SetSize(nedof);

        //const int eos = elemIndex * nedof;
        const int eos = e * nedof;
        for (int j=0; j<rdim; ++j)
        {
            for (int k=0; k<nedof; ++k)
                w_e[k] = W_E_elems(eos + k, j);

            // Compute the inner product, on this element, with the j-th W vector.
            eqpFe[j] += (v_e * Vloc_force) * (shape * w_e);
        }  // Loop (j) over V basis vectors
    } // Loop (i) over EQP points

    MPI_Allreduce(MPI_IN_PLACE, eqpFe.GetData(), eqpFe.Size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

#ifdef EQP_NO_Minv
    Vector invMvE(rdim);
    invMeROM.Mult(eqpFe, invMvE);
    eqpFe = invMvE;
#endif
}

// TODO: remove
void ROM_Operator::ForceIntegratorEQP_FOM(Vector & rhs) const
{
    Vector res(basis->GetDimV());

    ForceIntegratorEQP(res);
    basis->LiftROMtoFOM_dVdt(res, rhs);
}

// TODO: remove
void ROM_Operator::ForceIntegratorEQP_E_FOM(Vector const& v, Vector & rhs) const
{
    Vector res(basis->GetDimE());

    ForceIntegratorEQP_E(v, res);
    basis->LiftROMtoFOM_dEdt(res, rhs);
}

void ROM_Operator::EQPmult(double t, hydrodynamics::LagrangianHydroOperator *oper,
                           Vector const& S, Vector &dS) const
{
    MFEM_VERIFY(S.Size() == basis->SolutionSize(), "");
    MFEM_VERIFY(dS.Size() == basis->SolutionSize(), "");

    // TODO: allocate all this data just once in the class.

    basis->LiftToSampleMesh(S, fx);

    oper->ResetQuadratureData();
    oper->SetTime(t);
    oper->Mult(fx, fy);  // fy is not computed and not used in the EQP case.

    const int rXsize = basis->GetDimX();
    const int rVsize = basis->GetDimV();
    const int rEsize = basis->GetDimE();
    MFEM_VERIFY(rXsize + rVsize + rEsize == basis->SolutionSize(), "");

    Vector rdv_dt, rdx_dt, rde_dt, rSv;

    rSv.SetDataAndSize(S.GetData() + rXsize, rVsize);

    rdx_dt.SetDataAndSize(dS.GetData(), rXsize);
    rdv_dt.SetDataAndSize(dS.GetData() + rXsize, rVsize);
    rde_dt.SetDataAndSize(dS.GetData() + rXsize + rVsize, rEsize);

    CAROM::Vector a(rSv.GetData(), rSv.Size(), false, false);
    CAROM::Vector b(rdx_dt.GetData(), rdx_dt.Size(), false, false);
    BXtBV->mult(a, b);

    b += *BXtV0;

    rdv_dt = eqpFv;
    rde_dt = eqpFe;
}

void ROM_Operator::StepRK2AvgEQP(Vector &S, double &t, double &dt) const
{
    MFEM_VERIFY(hyperreduce && hyperreductionSamplingType == eqp, "");
    MFEM_VERIFY(S.Size() == basis->SolutionSize(), "");

    operSP->SetRomOperator(this);

    basis->LiftToSampleMesh(S, fx);

    operSP->ResetQuadratureData();
    operSP->SetTime(t);

    // TODO: can SolveEnergy be skipped here? We just want SolveVelocity at this point,
    // as well as `UpdateMesh(S)` and `InitEQP`.
    // This Mult calls includes `UpdateMesh`.
    operSP->Mult(fx, fy);  // fy is not computed and not used in the EQP case.

    operSP->SetQuadDataCurrent();

    // TODO: allocate this just once in the class.
    Vector Shalf(S.Size());
    Shalf = S;

    const int rXsize = basis->GetDimX();
    const int rVsize = basis->GetDimV();
    const int rEsize = basis->GetDimE();
    MFEM_VERIFY(rXsize + rVsize + rEsize == basis->SolutionSize(), "");

    Vector Shv, Shx, She;
    Vector Sv, Sx, Se;
    Vector vbar(rVsize);

    Shx.SetDataAndSize(Shalf.GetData(), rXsize);
    Shv.SetDataAndSize(Shalf.GetData() + rXsize, rVsize);
    She.SetDataAndSize(Shalf.GetData() + rXsize + rVsize, rEsize);

    Sx.SetDataAndSize(S.GetData(), rXsize);
    Sv.SetDataAndSize(S.GetData() + rXsize, rVsize);
    Se.SetDataAndSize(S.GetData() + rXsize + rVsize, rEsize);

    vbar = Sv;

    Shv.Add(0.5 * dt, eqpFv);

    // Lift Shv to vh
    const int sizeH1sp = basis->SolutionSizeH1SP();
    Vector vh(sizeH1sp);

    basis->LiftToSampleMesh_V(Shv, vh);

    operSP->SolveEnergy(fx, vh, fy);  // fy is not computed and not used in the EQP case

    She.Add(0.5 * dt, eqpFe);

    Vector dx(rXsize);
    CAROM::Vector vbar_libROM(vbar.GetData(), rVsize, false, false);
    CAROM::Vector dx_libROM(dx.GetData(), rXsize, false, false);
    CAROM::Vector Shv_libROM(Shv.GetData(), rVsize, false, false);
    BXtBV->mult(Shv_libROM, dx_libROM);
    dx_libROM += *BXtV0;

    Shx.Add(0.5 * dt, dx);

    // Now Shalf = (Sx, Sv, Se) is set to the half step. Next, set S_{n+1}.
    basis->LiftToSampleMesh(Shalf, fx);
    operSP->ResetQuadratureData();
    // TODO: skip SolveEnergy here?
    operSP->Mult(fx, fy);  // fy is not computed and not used in the EQP case
    operSP->SetQuadDataCurrent();

    Sv.Add(dt, eqpFv);

    vbar += Sv;
    vbar *= 0.5;

    basis->LiftToSampleMesh_V(vbar, vh);

    operSP->SolveEnergy(fx, vh, fy);  // fy is not computed and not used in the EQP case

    Se.Add(dt, eqpFe);

    BXtBV->mult(vbar_libROM, dx_libROM);
    dx_libROM += *BXtV0;

    Sx.Add(dt, dx);

    operSP->ResetQuadratureData();
}
