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

#include "laghos_assembly.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem {
  namespace hydrodynamics {

    const Tensors1D *tensors1D = NULL;

    Tensors1D::Tensors1D(int H1order, int L2order, int nqp1D)
      : HQshape1D(H1order + 1, nqp1D),
        HQgrad1D(H1order + 1, nqp1D),
        LQshape1D(L2order + 1, nqp1D) {
      // In this miniapp we assume:
      // - Gauss-Legendre quadrature points.
      // - Gauss-Lobatto continuous kinematic basis.
      // - Bernstein discontinuous thermodynamic basis.

      const double *quad1D_pos = poly1d.GetPoints(nqp1D - 1,
                                                  Quadrature1D::GaussLegendre);
      Poly_1D::Basis &basisH1 = poly1d.GetBasis(H1order,
                                                Quadrature1D::GaussLobatto);
      Vector col, grad_col;
      for (int q = 0; q < nqp1D; q++)
        {
          HQshape1D.GetColumnReference(q, col);
          HQgrad1D.GetColumnReference(q, grad_col);
          basisH1.Eval(quad1D_pos[q], col, grad_col);
        }
      for (int q = 0; q < nqp1D; q++)
        {
          LQshape1D.GetColumnReference(q, col);
          poly1d.CalcBernstein(L2order, quad1D_pos[q], col);
        }
    }

    QuadratureData::QuadratureData(int dim,
                                   int elements,
                                   int nqp) {
      Setup(occa::getDevice(), dim, elements, nqp);
    }

    QuadratureData::QuadratureData(occa::device device_,
                                   int dim,
                                   int elements,
                                   int nqp) {
      Setup(device_, dim, elements, nqp);
    }

    void QuadratureData::Setup(occa::device device_,
                               int dim,
                               int elements,
                               int nqp) {
      device = device_;

      rho0DetJ0w.SetSize(device, nqp * elements);
      stressJinvT.SetSize(device, dim * dim * nqp * elements);
      dtEst.SetSize(device, nqp * elements);
    }

    void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                                   ElementTransformation &Tr,
                                                   const IntegrationRule &integ_rule,
                                                   Vector &rho0DetJ0w,
                                                   Vector &elvect) {
      const int ip_cnt = integ_rule.GetNPoints();
      Vector shape(fe.GetDof());

      elvect.SetSize(fe.GetDof());
      elvect = 0.0;

      for (int q = 0; q < ip_cnt; q++)
        {
          fe.CalcShape(integ_rule.IntPoint(q), shape);
          // Note that rhoDetJ = rho0DetJ0.
          shape *= rho0DetJ0w(Tr.ElementNo*ip_cnt + q);
          elvect += shape;
        }
    }

    OccaMassOperator::OccaMassOperator(OccaFiniteElementSpace &fes_,
                                       const IntegrationRule &integ_rule_,
                                       QuadratureData *quad_data_)
      : Operator(fes_.GetTrueVSize()),
        device(occa::getDevice()),
        fes(fes_),
        integ_rule(integ_rule_),
        bilinearForm(&fes),
        quad_data(quad_data_),
        x_gf(device, &fes),
        y_gf(device, &fes) {
      Setup();
    }

    OccaMassOperator::OccaMassOperator(occa::device device_,
                                       OccaFiniteElementSpace &fes_,
                                       const IntegrationRule &integ_rule_,
                                       QuadratureData *quad_data_)
      : Operator(fes_.GetTrueVSize()),
        device(device_),
        fes(fes_),
        integ_rule(integ_rule_),
        bilinearForm(&fes),
        quad_data(quad_data_),
        x_gf(device, &fes),
        y_gf(device, &fes) {
      Setup();
    }

    void OccaMassOperator::Setup() {
      dim = fes.GetMesh()->Dimension();
      elements = fes.GetMesh()->GetNE();

      ess_tdofs_count = 0;

      OccaMassIntegrator &massInteg = *(new OccaMassIntegrator());
      massInteg.SetIntegrationRule(integ_rule);
      massInteg.SetOperator(quad_data->rho0DetJ0w);

      bilinearForm.AddDomainIntegrator(&massInteg);
      bilinearForm.Assemble();

      bilinearForm.FormOperator(Array<int>(), massOperator);
    }

    void OccaMassOperator::SetEssentialTrueDofs(Array<int> &dofs) {
      ess_tdofs_count = dofs.Size();
      if (ess_tdofs_count == 0) {
        return;
      }
      if (ess_tdofs.size<int>() < ess_tdofs_count) {
        ess_tdofs = device.malloc(ess_tdofs_count * sizeof(int),
                                  dofs.GetData());
      } else {
        ess_tdofs.copyFrom(dofs.GetData(),
                           ess_tdofs_count * sizeof(int));
      }
    }

    void OccaMassOperator::Mult(const OccaVector &x, OccaVector &y) const {
      if (ess_tdofs_count) {
        distX = x;
        distX.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
        x_gf.Distribute(distX);
      } else {
        x_gf.Distribute(x);
      }

      massOperator->Mult(x_gf, y_gf);

      fes.GetProlongationOperator()->MultTranspose(y_gf, y);

      if (ess_tdofs_count) {
        y.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
      }
    }

    void OccaMassOperator::EliminateRHS(OccaVector &b) {
      if (ess_tdofs_count) {
        b.SetSubVector(ess_tdofs, 0.0, ess_tdofs_count);
      }
    }

    OccaForceOperator::OccaForceOperator(OccaFiniteElementSpace &h1fes_,
                                         OccaFiniteElementSpace &l2fes_,
                                         const IntegrationRule &integ_rule_,
                                         QuadratureData *quad_data_)
      : Operator(l2fes_.GetTrueVSize(), h1fes_.GetTrueVSize()),
        device(occa::getDevice()),
        h1fes(h1fes_),
        l2fes(l2fes_),
        integ_rule(integ_rule_),
        quad_data(quad_data_) {
      Setup();
    }

    OccaForceOperator::OccaForceOperator(occa::device device_,
                                         OccaFiniteElementSpace &h1fes_,
                                         OccaFiniteElementSpace &l2fes_,
                                         const IntegrationRule &integ_rule_,
                                         QuadratureData *quad_data_)
      : Operator(l2fes_.GetTrueVSize(), h1fes_.GetTrueVSize()),
        device(device_),
        h1fes(h1fes_),
        l2fes(l2fes_),
        integ_rule(integ_rule_),
        quad_data(quad_data_) {
      Setup();
    }

    void OccaForceOperator::Setup() {
      dim = h1fes.GetMesh()->Dimension();
      elements = h1fes.GetMesh()->GetNE();

      occa::properties h1Props, l2Props, props;
      SetProperties(h1fes, integ_rule, h1Props);
      SetProperties(l2fes, integ_rule, l2Props);

      props = h1Props;
      props["defines/L2_DOFS_1D"] = l2Props["defines/NUM_DOFS_1D"];
      props["defines/H1_DOFS_1D"] = h1Props["defines/NUM_DOFS_1D"];

      multKernel = device.buildKernel("occa://laghos/force.okl",
                                      stringWithDim("Mult", dim),
                                      props);

      multTransposeKernel = device.buildKernel("occa://laghos/force.okl",
                                               stringWithDim("MultTranspose", dim),
                                               props);

      h1D2Q = OccaDofQuadMaps::Get(device, h1fes, integ_rule);
      l2D2Q = OccaDofQuadMaps::Get(device, l2fes, integ_rule);

      if (!tensors1D) {
        tensors1D = new Tensors1D(h1fes.GetFE(0)->GetOrder(),
                                  l2fes.GetFE(0)->GetOrder(),
                                  int(floor(0.7 + pow(integ_rule.GetNPoints(), 1.0 / dim))));
      }
    }

    void OccaForceOperator::Mult(const OccaVector &vecL2, OccaVector &vecH1) const {
      OccaVector gVecL2(device,
                        l2fes.GetLocalDofs() * elements);
      OccaVector gVecH1(device,
                        h1fes.GetVDim() * h1fes.GetLocalDofs() * elements);

      l2fes.GlobalToLocal(vecL2, gVecL2);

      multKernel(elements,
                 l2D2Q.dofToQuad,
                 h1D2Q.quadToDof,
                 h1D2Q.quadToDofD,
                 quad_data->stressJinvT,
                 gVecL2,
                 gVecH1);

      h1fes.LocalToGlobal(gVecH1, vecH1);
    }

    void OccaForceOperator::MultTranspose(const OccaVector &vecH1, OccaVector &vecL2) const {
      if (dim == 2) {
        OccaVector gVecH1(device,
                          h1fes.GetVDim() * h1fes.GetLocalDofs() * elements);
        OccaVector gVecL2(device,
                          l2fes.GetLocalDofs() * elements);

        h1fes.GlobalToLocal(vecH1, gVecH1);

        multTransposeKernel(elements,
                            l2D2Q.quadToDof,
                            h1D2Q.dofToQuad,
                            h1D2Q.dofToQuadD,
                            quad_data->stressJinvT,
                            gVecH1,
                            gVecL2);

        l2fes.LocalToGlobal(gVecL2, vecL2);
      } else {
        Vector vecH1_ = vecH1;
        Vector vecL2_ = vecL2;
        MultTransposeHex(vecH1_, vecL2_);
        vecL2 = vecL2_;
      }
    }


    // Force matrix action on hexahedral elements in 3D
    void OccaForceOperator::MultHex(const Vector &vecL2, Vector &vecH1) const {
      FiniteElementSpace &H1FESpace = *(h1fes.GetFESpace());
      FiniteElementSpace &L2FESpace = *(l2fes.GetFESpace());

      const int nqp = integ_rule.GetNPoints();
      Vector stressJinvT_ = quad_data->stressJinvT;
      DenseTensor stressJinvT(elements * nqp, dim, dim);
      int o_idx = 0;
      for (int el = 0; el < elements; ++el) {
        for (int q = 0; q < nqp; ++q) {
          for (int j = 0; j < dim; ++j) {
            for (int i = 0; i < dim; ++i) {
              stressJinvT(q + el*nqp, j, i) = stressJinvT_[o_idx++];
            }
          }
        }
      }

      const int nH1dof1D = tensors1D->HQshape1D.Height(),
        nL2dof1D = tensors1D->LQshape1D.Height(),
        nqp1D    = tensors1D->HQshape1D.Width(),
        nH1dof   = nH1dof1D * nH1dof1D * nH1dof1D;
      Array<int> h1dofs, l2dofs;

      Vector e(nL2dof1D * nL2dof1D * nL2dof1D);
      DenseMatrix E(e.GetData(), nL2dof1D*nL2dof1D, nL2dof1D);

      DenseMatrix HH_Q(nH1dof1D * nH1dof1D, nqp1D),
        H_HQ(HH_Q.GetData(), nH1dof1D, nH1dof1D*nqp1D),
        Q_HQ(nqp1D, nH1dof1D*nqp1D);
      DenseMatrix LL_Q(nL2dof1D * nL2dof1D, nqp1D),
        L_LQ(LL_Q.GetData(), nL2dof1D, nL2dof1D*nqp1D),
        Q_LQ(nqp1D, nL2dof1D*nqp1D);
      DenseMatrix QQ_Q(nqp1D * nqp1D, nqp1D), QQ_Qc(nqp1D * nqp1D, nqp1D);
      double *qqq = QQ_Q.GetData(), *qqqc = QQ_Qc.GetData();
      DenseMatrix HHHx(nH1dof1D * nH1dof1D, nH1dof1D),
        HHHy(nH1dof1D * nH1dof1D, nH1dof1D),
        HHHz(nH1dof1D * nH1dof1D, nH1dof1D);

      const H1_HexahedronElement *fe =
        dynamic_cast<const H1_HexahedronElement *>(H1FESpace.GetFE(0));
      const Array<int> &dof_map = fe->GetDofMap();

      vecH1 = 0.0;
      for (int z = 0; z < elements; z++) {
        // Note that the local numbering for L2 is the tensor numbering.
        L2FESpace.GetElementDofs(z, l2dofs);
        vecL2.GetSubVector(l2dofs, e);

        // LLQ_j1_j2_k3  = E_j1_j2_j3 LQs_j3_k3   -- contract in z direction.
        // QLQ_k1_j2_k3  = LQs_j1_k1 LLQ_j1_j2_k3 -- contract in x direction.
        // QQQ_k1_k2_k3  = QLQ_k1_j2_k3 LQs_j2_k2 -- contract in y direction.
        // The last step does some reordering (it's not product of matrices).
        mfem::Mult(E, tensors1D->LQshape1D, LL_Q);
        MultAtB(tensors1D->LQshape1D, L_LQ, Q_LQ);
        for (int k1 = 0; k1 < nqp1D; k1++) {
          for (int k2 = 0; k2 < nqp1D; k2++) {
            for (int k3 = 0; k3 < nqp1D; k3++) {
              QQ_Q(k1 + nqp1D*k2, k3) = 0.0;
              for (int j2 = 0; j2 < nL2dof1D; j2++) {
                QQ_Q(k1 + nqp1D*k2, k3) +=
                  Q_LQ(k1, j2 + k3*nL2dof1D) * tensors1D->LQshape1D(j2, k2);
              }
            }
          }
        }

        // Iterate over the components (x, y, z) of the result.
        for (int c = 0; c < 3; c++) {
          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,0) -- stress scaling d[v_c]_dx.
          double *d = stressJinvT(c).GetData() + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] = qqq[q] * d[q];
          };

          // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQs_i2_k2 -- contract  in y direction.
          // This first step does some reordering (it's not product of matrices).
          // HHQ_i1_i2_k3  = HQg_i1_k1 QHQ_k1_i2_k3  -- gradients in x direction.
          // HHHx_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- contract  in z direction.
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int i2 = 0; i2 < nH1dof1D; i2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                for (int k2 = 0; k2 < nqp1D; k2++) {
                  Q_HQ(k1, i2 + nH1dof1D*k3) +=
                    QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQshape1D(i2, k2);
                }
              }
            }
          }
          mfem::Mult(tensors1D->HQgrad1D, Q_HQ, H_HQ);
          MultABt(HH_Q, tensors1D->HQshape1D, HHHx);

          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,1) -- stress scaling d[v_c]_dy.
          d = stressJinvT(c).GetData() + 1*elements*nqp + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] = qqq[q] * d[q];
          };

          // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQg_i2_k2 -- gradients in y direction.
          // This first step does some reordering (it's not product of matrices).
          // HHQ_i1_i2_k3  = HQs_i1_k1 QHQ_k1_i2_k3  -- contract  in x direction.
          // HHHy_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- contract  in z direction.
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int i2 = 0; i2 < nH1dof1D; i2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                for (int k2 = 0; k2 < nqp1D; k2++) {
                  Q_HQ(k1, i2 + nH1dof1D*k3) +=
                    QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQgrad1D(i2, k2);
                }
              }
            }
          }
          mfem::Mult(tensors1D->HQshape1D, Q_HQ, H_HQ);
          MultABt(HH_Q, tensors1D->HQshape1D, HHHy);

          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,2) -- stress scaling d[v_c]_dz.
          d = stressJinvT(c).GetData() + 2*elements*nqp + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] = qqq[q] * d[q];
          };

          // QHQ_k1_i2_k3  = QQQc_k1_k2_k3 HQg_i2_k2 -- contract  in y direction.
          // This first step does some reordering (it's not product of matrices).
          // HHQ_i1_i2_k3  = HQs_i1_k1 QHQ_k1_i2_k3  -- contract  in x direction.
          // HHHz_i1_i2_i3 = HHQ_i1_i2_k3 HQs_i3_k3  -- gradients in z direction.
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int i2 = 0; i2 < nH1dof1D; i2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                Q_HQ(k1, i2 + nH1dof1D*k3) = 0.0;
                for (int k2 = 0; k2 < nqp1D; k2++) {
                  Q_HQ(k1, i2 + nH1dof1D*k3) +=
                    QQ_Qc(k1 + nqp1D*k2, k3) * tensors1D->HQshape1D(i2, k2);
                }
              }
            }
          }
          mfem::Mult(tensors1D->HQshape1D, Q_HQ, H_HQ);
          MultABt(HH_Q, tensors1D->HQgrad1D, HHHz);

          // Set the c-component of the result.
          H1FESpace.GetElementVDofs(z, h1dofs);
          for (int i1 = 0; i1 < nH1dof1D; i1++) {
            for (int i2 = 0; i2 < nH1dof1D; i2++) {
              for (int i3 = 0; i3 < nH1dof1D; i3++) {
                // Transfer from the mfem's H1 local numbering to the tensor
                // structure numbering.
                const int idx = i3*nH1dof1D*nH1dof1D + i2*nH1dof1D + i1;
                vecH1[h1dofs[c*nH1dof + dof_map[idx]]] +=
                  HHHx(i1 + i2*nH1dof1D, i3) +
                  HHHy(i1 + i2*nH1dof1D, i3) +
                  HHHz(i1 + i2*nH1dof1D, i3);
              }
            }
          }
        }
      }
    }


    // Transpose force matrix action on hexahedral elements in 3D
    void OccaForceOperator::MultTransposeHex(const Vector &vecH1, Vector &vecL2) const {
      FiniteElementSpace &H1FESpace = *(h1fes.GetFESpace());
      FiniteElementSpace &L2FESpace = *(l2fes.GetFESpace());

      const int nqp = integ_rule.GetNPoints();
      Vector stressJinvT_ = quad_data->stressJinvT;
      DenseTensor stressJinvT(elements * nqp, dim, dim);
      int o_idx = 0;
      for (int el = 0; el < elements; ++el) {
        for (int q = 0; q < nqp; ++q) {
          for (int j = 0; j < dim; ++j) {
            for (int i = 0; i < dim; ++i) {
              stressJinvT(q + el*nqp, j, i) = stressJinvT_[o_idx++];
            }
          }
        }
      }

      const int nH1dof1D = tensors1D->HQshape1D.Height(),
        nL2dof1D = tensors1D->LQshape1D.Height(),
        nqp1D    = tensors1D->HQshape1D.Width(),
        nH1dof   = nH1dof1D * nH1dof1D * nH1dof1D;
      Array<int> h1dofs, l2dofs;

      Vector v(nH1dof * 3), e(nL2dof1D * nL2dof1D * nL2dof1D);
      DenseMatrix V, E(e.GetData(), nL2dof1D * nL2dof1D, nL2dof1D);

      DenseMatrix HH_Q(nH1dof1D * nH1dof1D, nqp1D),
        H_HQ(HH_Q.GetData(), nH1dof1D, nH1dof1D * nqp1D),
        Q_HQ(nqp1D, nH1dof1D*nqp1D);
      DenseMatrix LL_Q(nL2dof1D * nL2dof1D, nqp1D),
        L_LQ(LL_Q.GetData(), nL2dof1D, nL2dof1D * nqp1D),
        Q_LQ(nqp1D, nL2dof1D*nqp1D);
      DenseMatrix QQ_Q(nqp1D * nqp1D, nqp1D),  QQ_Qc(nqp1D * nqp1D, nqp1D);
      double *qqqc = QQ_Qc.GetData();

      const H1_HexahedronElement *fe =
        dynamic_cast<const H1_HexahedronElement *>(H1FESpace.GetFE(0));
      const Array<int> &dof_map = fe->GetDofMap();

      for (int z = 0; z < elements; z++) {
        H1FESpace.GetElementVDofs(z, h1dofs);

        // Form (stress:grad_v) at all quadrature points.
        QQ_Q = 0.0;
        for (int c = 0; c < 3; c++) {
          // Transfer from the mfem's H1 local numbering to the tensor structure
          // numbering.
          for (int j = 0; j < nH1dof; j++) {
            v[c*nH1dof + j] = vecH1[h1dofs[c*nH1dof + dof_map[j]]];
          }
          // Connect to [v_c], i.e., the c-component of v.
          V.UseExternalData(v.GetData() + c*nH1dof, nH1dof1D*nH1dof1D, nH1dof1D);

          // HHQ_i1_i2_k3  = V_i1_i2_i3 HQs_i3_k3   -- contract  in z direction.
          // QHQ_k1_i2_k3  = HQg_i1_k1 HHQ_i1_i2_k3 -- gradients in x direction.
          // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQs_i2_k2 -- contract  in y direction.
          // The last step does some reordering (it's not product of matrices).
          mfem::Mult(V, tensors1D->HQshape1D, HH_Q);
          MultAtB(tensors1D->HQgrad1D, H_HQ, Q_HQ);
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                for (int i2 = 0; i2 < nH1dof1D; i2++) {
                  QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                    tensors1D->HQshape1D(i2, k2);
                }
              }
            }
          }
          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,0) -- stress scaling d[v_c]_dx.
          double *d = stressJinvT(c).GetData() + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] *= d[q];
          };
          // Add the (stress(c,0) * d[v_c]_dx) part of (stress:grad_v).
          QQ_Q += QQ_Qc;

          // HHQ_i1_i2_k3  = V_i1_i2_i3 HQs_i3_k3   -- contract  in z direction.
          // QHQ_k1_i2_k3  = HQs_i1_k1 HHQ_i1_i2_k3 -- contract  in x direction.
          // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQg_i2_k2 -- gradients in y direction.
          // The last step does some reordering (it's not product of matrices).
          mfem::Mult(V, tensors1D->HQshape1D, HH_Q);
          MultAtB(tensors1D->HQshape1D, H_HQ, Q_HQ);
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                for (int i2 = 0; i2 < nH1dof1D; i2++) {
                  QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                    tensors1D->HQgrad1D(i2, k2);
                }
              }
            }
          }
          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,1) -- stress scaling d[v_c]_dy.
          d = stressJinvT(c).GetData() + 1*elements*nqp + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] *= d[q];
          };
          // Add the (stress(c,1) * d[v_c]_dy) part of (stress:grad_v).
          QQ_Q += QQ_Qc;

          // HHQ_i1_i2_k3  = V_i1_i2_i3 HQg_i3_k3   -- gradients in z direction.
          // QHQ_k1_i2_k3  = HQs_i1_k1 HHQ_i1_i2_k3 -- contract  in x direction.
          // QQQc_k1_k2_k3 = QHQ_k1_i2_k3 HQs_i2_k2 -- contract  in y direction.
          // The last step does some reordering (it's not product of matrices).
          mfem::Mult(V, tensors1D->HQgrad1D, HH_Q);
          MultAtB(tensors1D->HQshape1D, H_HQ, Q_HQ);
          for (int k1 = 0; k1 < nqp1D; k1++) {
            for (int k2 = 0; k2 < nqp1D; k2++) {
              for (int k3 = 0; k3 < nqp1D; k3++) {
                QQ_Qc(k1 + nqp1D*k2, k3) = 0.0;
                for (int i2 = 0; i2 < nH1dof1D; i2++) {
                  QQ_Qc(k1 + nqp1D*k2, k3) += Q_HQ(k1, i2 + k3*nH1dof1D) *
                    tensors1D->HQshape1D(i2, k2);
                }
              }
            }
          }
          // QQQc_k1_k2_k3 *= stress_k1_k2_k3(c,2) -- stress scaling d[v_c]_dz.
          d = stressJinvT(c).GetData() + 2*elements*nqp + z*nqp;
          for (int q = 0; q < nqp; q++) {
            qqqc[q] *= d[q];
          };
          // Add the (stress(c,2) * d[v_c]_dz) part of (stress:grad_v).
          QQ_Q += QQ_Qc;
        }

        // QLQ_k1_j2_k3 = QQQ_k1_k2_k3 LQs_j2_k2 -- contract in y direction.
        // This first step does some reordering (it's not product of matrices).
        // LLQ_j1_j2_k3 = LQs_j1_k1 QLQ_k1_j2_k3 -- contract in x direction.
        // E_j1_j2_i3   = LLQ_j1_j2_k3 LQs_j3_k3 -- contract in z direction.
        for (int k1 = 0; k1 < nqp1D; k1++) {
          for (int j2 = 0; j2 < nL2dof1D; j2++) {
            for (int k3 = 0; k3 < nqp1D; k3++) {
              Q_LQ(k1, j2 + nL2dof1D*k3) = 0.0;
              for (int k2 = 0; k2 < nqp1D; k2++) {
                Q_LQ(k1, j2 + nL2dof1D*k3) +=
                  QQ_Q(k1 + nqp1D*k2, k3) * tensors1D->LQshape1D(j2, k2);
              }
            }
          }
        }
        mfem::Mult(tensors1D->LQshape1D, Q_LQ, L_LQ);
        MultABt(LL_Q, tensors1D->LQshape1D, E);

        L2FESpace.GetElementDofs(z, l2dofs);
        vecL2.SetSubVector(l2dofs, e);
      }
    }
  } // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
