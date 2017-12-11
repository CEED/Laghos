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
#include "defines.hpp"

// *****************************************************************************
static void rIniGeom1D(const int NUM_DOFS,
                       const int NUM_QUAD,
                       const int numElements,
                       const double* __restrict dofToQuadD,
                       const double* __restrict nodes,
                       double* __restrict J,
                       double* __restrict invJ,
                       double* __restrict detJ) {
  assert(NUM_DOFS==4); const int nd = 4;
  
  forall(numElements,[=]_device_(int e) {
    double s_nodes[nd];

    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d += NUM_QUAD) {
        s_nodes[d] = nodes[ijkN(0,d,e,NUM_QUAD)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijN(q,d,NUM_DOFS)];
        J11 += wx * s_nodes[d];
      }
      J[ijN(q,e,NUM_QUAD)] = J11;
      invJ[ijN(q, e,NUM_QUAD)] = 1.0 / J11;
      detJ[ijN(q, e,NUM_QUAD)] = J11;
    }
  });
}

// *****************************************************************************
static void rIniGeom2D(const int NUM_DOFS,
                       const int NUM_QUAD,
                       const int numElements,
                       const double* __restrict dofToQuadD,
                       const double* __restrict nodes,
                       double* __restrict J,
                       double* __restrict invJ,
                       double* __restrict detJ) {
  //printf("\033[31m[NUM_DOFS=%d]\033[m\n",NUM_DOFS);
  assert(NUM_DOFS==9); const int nd = 9;
  
  forall(numElements,[=]_device_(int e) {
    double s_nodes[2 * nd] ;
    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d +=NUM_QUAD) {
        s_nodes[ijN(0,d,2)] = nodes[ijkNM(0,d,e,2,NUM_DOFS)];
        s_nodes[ijN(1,d,2)] = nodes[ijkNM(1,d,e,2,NUM_DOFS)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0, J12 = 0;
      double J21 = 0, J22 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
        const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
        const double x = s_nodes[ijN(0,d,2)];
        const double y = s_nodes[ijN(1,d,2)];
        J11 += (wx * x); J12 += (wx * y);
        J21 += (wy * x); J22 += (wy * y);
      }
      const double r_detJ = (J11 * J22)-(J12 * J21);
      J[ijklNM(0, 0, q, e,2,NUM_QUAD)] = J11;
      J[ijklNM(1, 0, q, e,2,NUM_QUAD)] = J12;
      J[ijklNM(0, 1, q, e,2,NUM_QUAD)] = J21;
      J[ijklNM(1, 1, q, e,2,NUM_QUAD)] = J22;
      const double r_idetJ = 1.0 / r_detJ;
      invJ[ijklNM(0, 0, q, e,2,NUM_QUAD)] =  J22 * r_idetJ;
      invJ[ijklNM(1, 0, q, e,2,NUM_QUAD)] = -J12 * r_idetJ;

      invJ[ijklNM(0, 1, q, e,2,NUM_QUAD)] = -J21 * r_idetJ;
      invJ[ijklNM(1, 1, q, e,2,NUM_QUAD)] =  J11 * r_idetJ;
      detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
    }
  });
}

// *****************************************************************************
static void rIniGeom3D(const int NUM_DOFS,
                       const int NUM_QUAD,
                       const int numElements,
                       const double* __restrict dofToQuadD,
                       const double* __restrict nodes,
                       double* __restrict J,
                       double* __restrict invJ,
                       double* __restrict detJ) {
  assert(NUM_DOFS==4); const int nd = 4;
  forall(numElements,[=]_device_(int e) {
    double s_nodes[3 * nd] ;
    for (int q = 0; q < NUM_QUAD; ++q) {
      for (int d = q; d < NUM_DOFS; d += NUM_QUAD) {
        s_nodes[ijN(0,d,3)] = nodes[ijkNM(0, d, e,3,NUM_DOFS)];
        s_nodes[ijN(1,d,3)] = nodes[ijkNM(1, d, e,3,NUM_DOFS)];
        s_nodes[ijN(2,d,3)] = nodes[ijkNM(2, d, e,3,NUM_DOFS)];
      }
    }
    for (int q = 0; q < NUM_QUAD; ++q) {
      double J11 = 0, J12 = 0, J13 = 0;
      double J21 = 0, J22 = 0, J23 = 0;
      double J31 = 0, J32 = 0, J33 = 0;
      for (int d = 0; d < NUM_DOFS; ++d) {
        const double wx = dofToQuadD[ijkNM(0, q, d,3,NUM_QUAD)];
        const double wy = dofToQuadD[ijkNM(1, q, d,3,NUM_QUAD)];
        const double wz = dofToQuadD[ijkNM(2, q, d,3,NUM_QUAD)];
        const double x = s_nodes[ijN(0, d,3)];
        const double y = s_nodes[ijN(1, d,3)];
        const double z = s_nodes[ijN(2, d,3)];
        J11 += (wx * x); J12 += (wx * y); J13 += (wx * z);
        J21 += (wy * x); J22 += (wy * y); J23 += (wy * z);
        J31 += (wz * x); J32 += (wz * y); J33 += (wz * z);
      }
      const double r_detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                             (J13 * J21 * J32) -
                             (J13 * J22 * J31)-(J12 * J21 * J33)-(J11 * J23 * J32));
      J[ijklNM(0, 0, q, e,3,NUM_QUAD)] = J11;
      J[ijklNM(1, 0, q, e,3,NUM_QUAD)] = J12;
      J[ijklNM(2, 0, q, e,3,NUM_QUAD)] = J13;
      J[ijklNM(0, 1, q, e,3,NUM_QUAD)] = J21;
      J[ijklNM(1, 1, q, e,3,NUM_QUAD)] = J22;
      J[ijklNM(2, 1, q, e,3,NUM_QUAD)] = J23;
      J[ijklNM(0, 2, q, e,3,NUM_QUAD)] = J31;
      J[ijklNM(1, 2, q, e,3,NUM_QUAD)] = J32;
      J[ijklNM(2, 2, q, e,3,NUM_QUAD)] = J33;

      const double r_idetJ = 1.0 / r_detJ;
      invJ[ijklNM(0, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J22 * J33)-(J23 * J32));
      invJ[ijklNM(1, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J32 * J13)-(J33 * J12));
      invJ[ijklNM(2, 0, q, e,3,NUM_QUAD)] = r_idetJ * ((J12 * J23)-(J13 * J22));

      invJ[ijklNM(0, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J23 * J31)-(J21 * J33));
      invJ[ijklNM(1, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J33 * J11)-(J31 * J13));
      invJ[ijklNM(2, 1, q, e,3,NUM_QUAD)] = r_idetJ * ((J13 * J21)-(J11 * J23));

      invJ[ijklNM(0, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J21 * J32)-(J22 * J31));
      invJ[ijklNM(1, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J31 * J12)-(J32 * J11));
      invJ[ijklNM(2, 2, q, e,3,NUM_QUAD)] = r_idetJ * ((J11 * J22)-(J12 * J21));
      detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
    }
  });
}

// *****************************************************************************
void rIniGeom(const int dim,
              const int NUM_DOFS,
              const int NUM_QUAD,
              const int numElements,
              const double* dofToQuadD,
              const double* nodes,
              double* __restrict J,
              double* __restrict invJ,
              double* __restrict detJ) {
  switch (dim) {
    case 1: {rIniGeom1D(NUM_DOFS,NUM_QUAD,numElements,dofToQuadD,nodes,J,invJ,detJ); break;}
    case 2: {rIniGeom2D(NUM_DOFS,NUM_QUAD,numElements,dofToQuadD,nodes,J,invJ,detJ); break;}
    case 3: {rIniGeom3D(NUM_DOFS,NUM_QUAD,numElements,dofToQuadD,nodes,J,invJ,detJ); break;}
    default:assert(false);
  }
}
