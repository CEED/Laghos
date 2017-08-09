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

namespace mfem
{

namespace hydrodynamics
{

const Tensors1D *tensors1D = NULL;

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

  Jac0inv.SetSize(dim, dim, elements * nqp);
  stressJinvT.SetSize(elements * nqp, dim, dim);
  rho0DetJ0w.SetSize(elements * nqp);

  o_rho0DetJ0w.SetSize(device, nqp * elements);
  o_stressJinvT.SetSize(device, dim * dim * nqp * elements);
  o_dtEst.SetSize(device, nqp * elements);
}

Tensors1D::Tensors1D(int H1order, int L2order, int nqp1D)
   : HQshape1D(H1order + 1, nqp1D),
     HQgrad1D(H1order + 1, nqp1D),
     LQshape1D(L2order + 1, nqp1D)
{
   // In this miniapp we assume:
   // - Gauss-Legendre quadrature points.
   // - Gauss-Lobatto continuous kinematic basis.
   // - Bernstein discontinuous thermodynamic basis.

   const double *quad1D_pos = poly1d.GetPoints(nqp1D - 1,
                                               BasisType::GaussLegendre);
   Poly_1D::Basis &basisH1 = poly1d.GetBasis(H1order,
                                             BasisType::GaussLobatto);
   Poly_1D::Basis &basisL2 = poly1d.GetBasis(L2order,
                                             BasisType::Positive,
                                             Poly_1D::Positive);

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
      basisL2.Eval(quad1D_pos[q], col);
   }
}

void DensityIntegrator::AssembleRHSElementVect(const FiniteElement &fe,
                                               ElementTransformation &Tr,
                                               Vector &elvect)
{
   const int ip_cnt = IntRule->GetNPoints();
   Vector shape(fe.GetDof());

   elvect.SetSize(fe.GetDof());
   elvect = 0.0;

   for (int q = 0; q < ip_cnt; q++)
   {
      fe.CalcShape(IntRule->IntPoint(q), shape);
      // Note that rhoDetJ = rho0DetJ0.
      shape *= quad_data.rho0DetJ0w(Tr.ElementNo*ip_cnt + q);
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

  OccaCoefficient coeff("(rho0DetJ0w(q, e) / (quadWeights[q] * detJ))");
  coeff.AddVector("rho0DetJ0w",
                  quad_data->o_rho0DetJ0w,
                  "@dim(NUM_QUAD, numElements)",
                  true);

  OccaMassIntegrator &massInteg = *(new OccaMassIntegrator(coeff));
  massInteg.SetIntegrationRule(integ_rule);

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
                                  "Mult2D",
                                  props);

  h1D2Q = OccaDofQuadMaps::Get(device, h1fes, integ_rule);
  l2D2Q = OccaDofQuadMaps::Get(device, l2fes, integ_rule);
}

void OccaForceOperator::Mult(const OccaVector &vecL2, OccaVector &vecH1) const {
  if ((dim == 2) && l2fes.hasTensorBasis()) {
    OccaVector gVecH1(device,
                      h1fes.GetVDim() * h1fes.GetLocalDofs() * elements);

    multKernel(elements,
               l2D2Q.dofToQuad,
               h1D2Q.quadToDof,
               h1D2Q.quadToDofD,
               quad_data->o_stressJinvT,
               vecL2,
               gVecH1);

    h1fes.LocalToGlobal(gVecH1, vecH1);

    Vector v1 = vecH1;
    Vector v2(v1.Size());
    const int h1GlobalDofs = h1fes.GetGlobalDofs();
    for (int c = 0; c < 2; ++c) {
      for (int d = 0; d < h1GlobalDofs; ++d) {
        v2[d + c*h1GlobalDofs] = v1[c + d*2];
      }
    }
    vecH1 = v2;

    return;
  }

  if (dim == 2) {
    MultQuad(vecL2, vecH1);
  } else if (dim == 3) {
    MultHex(vecL2, vecH1);
  } else {
    MFEM_ABORT("Unsupported dimension");
  }
}

void OccaForceOperator::MultTranspose(const OccaVector &vecH1, OccaVector &vecL2) const {
  if (dim == 2) {
    MultTransposeQuad(vecH1, vecL2);
  } else if (dim == 3) {
    MultTransposeHex(vecH1, vecL2);
  } else {
    MFEM_ABORT("Unsupported dimension");
  }
}

void OccaForceOperator::MultQuad(const OccaVector &vecL2, OccaVector &vecH1) const {
  Vector h_vecL2 = vecL2;
  Vector h_vecH1;

  const int nH1dof1D = tensors1D->HQshape1D.Height();
  const int nL2dof1D = tensors1D->LQshape1D.Height();
  const int nqp1D    = tensors1D->HQshape1D.Width();
  const int nqp      =  nqp1D * nqp1D;
  const int nH1dof   = nH1dof1D * nH1dof1D;

  Array<int> h1dofs, l2dofs;
  Vector e(nL2dof1D * nL2dof1D);

  DenseMatrix E(e.GetData(), nL2dof1D, nL2dof1D);
  DenseMatrix LQ(nL2dof1D, nqp1D);
  DenseMatrix HQ(nH1dof1D, nqp1D);
  DenseMatrix QQ(nqp1D, nqp1D);
  DenseMatrix HHx(nH1dof1D, nH1dof1D);
  DenseMatrix HHy(nH1dof1D, nH1dof1D);

  // Quadrature data for a specific direction.
  DenseMatrix QQd(nqp1D, nqp1D);
  double *data_qd = QQd.GetData(), *data_q = QQ.GetData();

  const H1_QuadrilateralElement *fe =
    dynamic_cast<const H1_QuadrilateralElement *>(h1fes.GetFE(0));
  const Array<int> &dof_map = fe->GetDofMap();

  h_vecH1.SetSize(vecH1.Size());
  h_vecH1 = 0.0;

  for (int el = 0; el < elements; el++) {
    // Note that the local numbering for L2 is the tensor numbering.
    l2fes.GetFESpace()->GetElementDofs(el, l2dofs);
    h_vecL2.GetSubVector(l2dofs, e);

    // LQ_j2_k1 = E_j1_j2 LQs_j1_k1  -- contract in x direction.
    // QQ_k1_k2 = LQ_j2_k1 LQs_j2_k2 -- contract in y direction.
    MultAtB(E, tensors1D->LQshape1D, LQ);
    MultAtB(LQ, tensors1D->LQshape1D, QQ);

    // Iterate over the components (x and y) of the result.
    for (int c = 0; c < 2; c++) {
      // QQd_k1_k2 *= stress_k1_k2(c,0)  -- stress that scales d[v_c]_dx.
      // HQ_i2_k1   = HQs_i2_k2 QQ_k1_k2 -- contract in y direction.
      // HHx_i1_i2  = HQg_i1_k1 HQ_i2_k1 -- gradients in x direction.
      double *d = quad_data->stressJinvT(c).GetData() + el*nqp;
      for (int q = 0; q < nqp; q++) {
        data_qd[q] = data_q[q] * d[q];
      };
      MultABt(tensors1D->HQshape1D, QQd, HQ);
      MultABt(tensors1D->HQgrad1D, HQ, HHx);

      // QQd_k1_k2 *= stress_k1_k2(c,1) -- stress that scales d[v_c]_dy.
      // HQ_i2_k1  = HQg_i2_k2 QQ_k1_k2 -- gradients in y direction.
      // HHy_i1_i2 = HQ_i1_k1 HQ_i2_k1  -- contract in x direction.
      d = quad_data->stressJinvT(c).GetData() + 1*elements*nqp + el*nqp;
      for (int q = 0; q < nqp; q++) {
        data_qd[q] = data_q[q] * d[q];
      };
      MultABt(tensors1D->HQgrad1D, QQd, HQ);
      MultABt(tensors1D->HQshape1D, HQ, HHy);

      // Set the c-component of the result.
      h1fes.GetFESpace()->GetElementVDofs(el, h1dofs);
      for (int i1 = 0; i1 < nH1dof1D; i1++) {
        for (int i2 = 0; i2 < nH1dof1D; i2++) {
          // Transfer from the mfem's H1 local numbering to the tensor
          // structure numbering.
          const int idx = i2 * nH1dof1D + i1;
          h_vecH1[h1dofs[c*nH1dof + dof_map[idx]]] +=
            HHx(i1, i2) + HHy(i1, i2);
        }
      }
    }
  }

  vecH1 = h_vecH1;
}

void OccaForceOperator::MultHex(const OccaVector &vecL2, OccaVector &vecH1) const {
  Vector h_vecL2 = vecL2;
  Vector h_vecH1;

  const int nH1dof1D = tensors1D->HQshape1D.Height();
  const int nL2dof1D = tensors1D->LQshape1D.Height();
  const int nqp1D    = tensors1D->HQshape1D.Width();
  const int nqp      = nqp1D * nqp1D * nqp1D;
  const int nH1dof   = nH1dof1D * nH1dof1D * nH1dof1D;

  Array<int> h1dofs, l2dofs;

  Vector e(nL2dof1D * nL2dof1D * nL2dof1D);
  DenseMatrix E(e.GetData(), nL2dof1D*nL2dof1D, nL2dof1D);

  DenseMatrix HH_Q(nH1dof1D * nH1dof1D, nqp1D);
  DenseMatrix H_HQ(HH_Q.GetData(), nH1dof1D, nH1dof1D*nqp1D);
  DenseMatrix Q_HQ(nqp1D, nH1dof1D*nqp1D);

  DenseMatrix LL_Q(nL2dof1D * nL2dof1D, nqp1D);
  DenseMatrix L_LQ(LL_Q.GetData(), nL2dof1D, nL2dof1D*nqp1D);
  DenseMatrix Q_LQ(nqp1D, nL2dof1D*nqp1D);

  DenseMatrix QQ_Q(nqp1D * nqp1D, nqp1D);
  DenseMatrix QQ_Qc(nqp1D * nqp1D, nqp1D);

  double *qqq = QQ_Q.GetData();
  double *qqqc = QQ_Qc.GetData();

  DenseMatrix HHHx(nH1dof1D * nH1dof1D, nH1dof1D);
  DenseMatrix HHHy(nH1dof1D * nH1dof1D, nH1dof1D);
  DenseMatrix HHHz(nH1dof1D * nH1dof1D, nH1dof1D);

  const H1_HexahedronElement *fe =
    dynamic_cast<const H1_HexahedronElement *>(h1fes.GetFE(0));
  const Array<int> &dof_map = fe->GetDofMap();

  h_vecH1.SetSize(vecH1.Size());
  h_vecH1 = 0.0;

  for (int el = 0; el < elements; el++) {
    // Note that the local numbering for L2 is the tensor numbering.
    l2fes.GetFESpace()->GetElementVDofs(el, l2dofs);
    h_vecL2.GetSubVector(l2dofs, e);

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
      double *d = quad_data->stressJinvT(c).GetData() + el*nqp;
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
      d = quad_data->stressJinvT(c).GetData() + 1*elements*nqp + el*nqp;
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
      d = quad_data->stressJinvT(c).GetData() + 2*elements*nqp + el*nqp;
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
      h1fes.GetFESpace()->GetElementVDofs(el, h1dofs);
      for (int i1 = 0; i1 < nH1dof1D; i1++) {
        for (int i2 = 0; i2 < nH1dof1D; i2++) {
          for (int i3 = 0; i3 < nH1dof1D; i3++) {
            // Transfer from the mfem's H1 local numbering to the tensor
            // structure numbering.
            const int idx = i3*nH1dof1D*nH1dof1D + i2*nH1dof1D + i1;
            h_vecH1[h1dofs[c*nH1dof + dof_map[idx]]] +=
              HHHx(i1 + i2*nH1dof1D, i3) +
              HHHy(i1 + i2*nH1dof1D, i3) +
              HHHz(i1 + i2*nH1dof1D, i3);
          }
        }
      }
    }
  }

  vecH1 = h_vecH1;
}

void OccaForceOperator::MultTransposeQuad(const OccaVector &vecH1, OccaVector &vecL2) const {
  Vector h_vecH1 = vecH1;
  Vector h_vecL2;

  const int nH1dof1D = tensors1D->HQshape1D.Height();
  const int nL2dof1D = tensors1D->LQshape1D.Height();
  const int nqp1D    = tensors1D->HQshape1D.Width();
  const int nqp      = nqp1D * nqp1D;
  const int nH1dof   = nH1dof1D * nH1dof1D;

  Array<int> h1dofs, l2dofs;

  Vector v(nH1dof * 2), e(nL2dof1D * nL2dof1D);
  DenseMatrix V, E(e.GetData(), nL2dof1D, nL2dof1D);

  DenseMatrix HQ(nH1dof1D, nqp1D);
  DenseMatrix LQ(nL2dof1D, nqp1D);
  DenseMatrix QQc(nqp1D, nqp1D);
  DenseMatrix QQ(nqp1D, nqp1D);
  double *qqc = QQc.GetData();

  const H1_QuadrilateralElement *fe =
    dynamic_cast<const H1_QuadrilateralElement *>(h1fes.GetFE(0));
  const Array<int> &dof_map = fe->GetDofMap();

  h_vecL2.SetSize(vecL2.Size());

  for (int el = 0; el < elements; el++) {
    h1fes.GetFESpace()->GetElementVDofs(el, h1dofs);

    // Form (stress:grad_v) at all quadrature points.
    QQ = 0.0;
    for (int c = 0; c < 2; c++) {
      // Transfer from the mfem's H1 local numbering to the tensor structure
      // numbering.
      for (int j = 0; j < nH1dof; j++) {
        v[c*nH1dof + j] = h_vecH1[h1dofs[c*nH1dof + dof_map[j]]];
      }
      // Connect to [v_c], i.e., the c-component of v.
      V.UseExternalData(v.GetData() + c*nH1dof, nH1dof1D, nH1dof1D);

      // HQ_i2_k1   = V_i1_i2 HQg_i1_k1  -- gradients in x direction.
      // QQc_k1_k2  = HQ_i2_k1 HQs_i2_k2 -- contract  in y direction.
      // QQc_k1_k2 *= stress_k1_k2(c,0)  -- stress that scales d[v_c]_dx.
      MultAtB(V, tensors1D->HQgrad1D, HQ);
      MultAtB(HQ, tensors1D->HQshape1D, QQc);
      double *d = quad_data->stressJinvT(c).GetData() + el*nqp;
      for (int q = 0; q < nqp; q++) {
        qqc[q] *= d[q];
      }
      // Add the (stress(c,0) * d[v_c]_dx) part of (stress:grad_v).
      QQ += QQc;

      // HQ_i2_k1   = V_i1_i2 HQs_i1_k1  -- contract  in x direction.
      // QQc_k1_k2  = HQ_i2_k1 HQg_i2_k2 -- gradients in y direction.
      // QQc_k1_k2 *= stress_k1_k2(c,1)  -- stress that scales d[v_c]_dy.
      MultAtB(V, tensors1D->HQshape1D, HQ);
      MultAtB(HQ, tensors1D->HQgrad1D, QQc);
      d = quad_data->stressJinvT(c).GetData() + 1*elements*nqp + el*nqp;
      for (int q = 0; q < nqp; q++) {
        qqc[q] *= d[q];
      }
      // Add the (stress(c,1) * d[v_c]_dy) part of (stress:grad_v).
      QQ += QQc;
    }

    // LQ_j1_k2 = LQs_j1_k1 QQ_k1_k2 -- contract in x direction.
    // E_j1_j2  = LQ_j1_k2 LQs_j2_k2 -- contract in y direction.
    mfem::Mult(tensors1D->LQshape1D, QQ, LQ);
    MultABt(LQ, tensors1D->LQshape1D, E);

    l2fes.GetFESpace()->GetElementDofs(el, l2dofs);
    h_vecL2.SetSubVector(l2dofs, e);
  }

  vecL2 = h_vecL2;
}

void OccaForceOperator::MultTransposeHex(const OccaVector &vecH1, OccaVector &vecL2) const {
  Vector h_vecH1 = vecH1;
  Vector h_vecL2;

  const int nH1dof1D = tensors1D->HQshape1D.Height(),
    nL2dof1D = tensors1D->LQshape1D.Height(),
    nqp1D    = tensors1D->HQshape1D.Width(),
    nqp      = nqp1D * nqp1D * nqp1D,
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
    dynamic_cast<const H1_HexahedronElement *>(h1fes.GetFE(0));
  const Array<int> &dof_map = fe->GetDofMap();

  h_vecL2.SetSize(vecL2.Size());

  for (int el = 0; el < elements; el++) {
    h1fes.GetFESpace()->GetElementVDofs(el, h1dofs);

    // Form (stress:grad_v) at all quadrature points.
    QQ_Q = 0.0;
    for (int c = 0; c < 3; c++) {
      // Transfer from the mfem's H1 local numbering to the tensor structure
      // numbering.
      for (int j = 0; j < nH1dof; j++) {
        v[c*nH1dof + j] = h_vecH1[h1dofs[c*nH1dof + dof_map[j]]];
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
      double *d = quad_data->stressJinvT(c).GetData() + el*nqp;
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
      d = quad_data->stressJinvT(c).GetData() + 1*elements*nqp + el*nqp;
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
      d = quad_data->stressJinvT(c).GetData() + 2*elements*nqp + el*nqp;
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

    l2fes.GetFESpace()->GetElementDofs(el, l2dofs);
    h_vecL2.SetSubVector(l2dofs, e);
  }

  vecL2 = h_vecL2;
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
