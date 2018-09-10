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

#include "../laghos_solver.hpp"
#include "qupdate.hpp"

#ifdef MFEM_USE_MPI

namespace mfem
{

namespace hydrodynamics
{

   // **************************************************************************
   static void qIniGeom2D( const int NUM_DOFS,
                           const int NUM_QUAD,
                           const int numElements,
                           const double* __restrict dofToQuadD,
                           const double* __restrict nodes,
                           double* __restrict J,
                           double* __restrict invJ,
                           double* __restrict detJ){
      for(int e=0;e<numElements;e+=1){
         double s_nodes[2 * NUM_DOFS];
         for (int q = 0; q < NUM_QUAD; ++q)
         {
            for (int d = q; d < NUM_DOFS; d +=NUM_QUAD)
            {
               s_nodes[ijN(0,d,2)] = nodes[ijkNM(0,d,e,2,NUM_DOFS)];
               s_nodes[ijN(1,d,2)] = nodes[ijkNM(1,d,e,2,NUM_DOFS)];
            }
         }
         for (int q = 0; q < NUM_QUAD; ++q)
         {
            double J11 = 0; double J12 = 0;
            double J21 = 0; double J22 = 0;
            for (int d = 0; d < NUM_DOFS; ++d)
            {
               const double wx = dofToQuadD[ijkNM(0,q,d,2,NUM_QUAD)];
               const double wy = dofToQuadD[ijkNM(1,q,d,2,NUM_QUAD)];
               const double x = s_nodes[ijN(0,d,2)];
               const double y = s_nodes[ijN(1,d,2)];
               J11 += (wx * x); J12 += (wx * y);
               J21 += (wy * x); J22 += (wy * y);
            }
            const double r_detJ = (J11 * J22)-(J12 * J21);
            J[ijklNM(0,0,q,e,2,NUM_QUAD)] = J11;
            J[ijklNM(1,0,q,e,2,NUM_QUAD)] = J12;
            J[ijklNM(0,1,q,e,2,NUM_QUAD)] = J21;
            J[ijklNM(1,1,q,e,2,NUM_QUAD)] = J22;
            const double r_idetJ = 1.0 / r_detJ;
            invJ[ijklNM(0,0,q,e,2,NUM_QUAD)] =  J22 * r_idetJ;
            invJ[ijklNM(1,0,q,e,2,NUM_QUAD)] = -J12 * r_idetJ;
            invJ[ijklNM(0,1,q,e,2,NUM_QUAD)] = -J21 * r_idetJ;
            invJ[ijklNM(1,1,q,e,2,NUM_QUAD)] =  J11 * r_idetJ;
            detJ[ijN(q, e,NUM_QUAD)] = r_detJ;
         }
      }
   }

   // **************************************************************************
   static void qIniGeom(const int DIM,
                        const int NUM_DOFS,
                        const int NUM_QUAD,
                        const int numElements,
                        const double* dofToQuadD,
                        const double* nodes,
                        double* J,
                        double* invJ,
                        double* detJ){
      push();
      assert (DIM==2);
      qIniGeom2D(NUM_DOFS, NUM_QUAD, numElements,
                 dofToQuadD, nodes, J, invJ, detJ);
      pop();
   }

// *****************************************************************************
static qGeometry *geom=NULL;

// ***************************************************************************
// * ~ qGeometry
// ***************************************************************************
qGeometry::~qGeometry()
{
   push();
   free(geom->meshNodes);
   free(geom->J);
   free(geom->invJ);
   free(geom->detJ);
   delete[] geom;
   pop();
}

// *****************************************************************************
qGeometry* qGeometry::Get(FiniteElementSpace& fes,
                          const IntegrationRule& ir)
{
   push();
   Mesh& mesh = *(fes.GetMesh());
   const bool geom_to_allocate = !geom;
   if (geom_to_allocate) {
      dbg("geom_to_allocate: new qGeometry");
      geom = new qGeometry();
   }
   if (!mesh.GetNodes()) { mesh.SetCurvature(1, false, -1, Ordering::byVDIM); }
   GridFunction& nodes = *(mesh.GetNodes());
   const mfem::FiniteElementSpace& fespace = *(nodes.FESpace());
   const mfem::FiniteElement& fe = *(fespace.GetFE(0));
   const int dims     = fe.GetDim();
   const int elements = fespace.GetNE();
   const int numDofs  = fe.GetDof();
   const int numQuad  = ir.GetNPoints();
   const bool orderedByNODES = (fespace.GetOrdering() == Ordering::byNODES);
   dbg("orderedByNODES: %s", orderedByNODES?"true":"false");
   
   if (orderedByNODES) {
      dbg("\033[7morderedByNODES, ReorderByVDim");
      ReorderByVDim(nodes);
   }
   const int asize = dims*numDofs*elements;
   mfem::Array<double> meshNodes(asize);
   const Table& e2dTable = fespace.GetElementToDofTable();
   const int* elementMap = e2dTable.GetJ();
   mfem::Array<int> eMap(numDofs*elements);
   {
      push();
      for (int e = 0; e < elements; ++e)
      {
         for (int d = 0; d < numDofs; ++d)
         {
            const int lid = d+numDofs*e;
            const int gid = elementMap[lid];
            eMap[lid]=gid;
            for (int v = 0; v < dims; ++v)
            {
               const int moffset = v+dims*lid;
               const int xoffset = v+dims*gid;
               meshNodes[moffset] = nodes[xoffset];
            }
         }
      }
      pop();
   }
   if (geom_to_allocate)
   {
      dbg("eMap & meshNodes allocate");
      geom->meshNodes.allocate(dims, numDofs, elements);
      geom->eMap.allocate(numDofs, elements);
   }
   {
      push();
      geom->meshNodes = meshNodes;
      geom->eMap = eMap;
      pop();
   }
   
   // Reorder the original gf back
   if (orderedByNODES) {
      dbg("Reorder the original gf back");
      ReorderByNodes(nodes);
   }
   
   if (geom_to_allocate)
   {
      dbg("geom_to_allocate: J, invJ & detJ, numQuad=%d",numQuad);
      geom->J.allocate(dims, dims, numQuad, elements);
      geom->invJ.allocate(dims, dims, numQuad, elements);
      geom->detJ.allocate(numQuad, elements);
   }

   const qDofQuadMaps* maps = qDofQuadMaps::GetSimplexMaps(fe, ir);
   assert(maps);
   {
      dbg("dims=%d, numDofs=%d, numQuad=%d, elements=%d",dims,numDofs,numQuad,elements);
      push(rIniGeom,SteelBlue);
      qIniGeom(dims,numDofs,numQuad,elements,
               maps->dofToQuadD,
               geom->meshNodes,
               geom->J,
               geom->invJ,
               geom->detJ);
      pop();
   }
   pop();
   return geom;
}

// ***************************************************************************
void qGeometry::ReorderByVDim(GridFunction& nodes)
{
   push();
   const mfem::FiniteElementSpace *fes=nodes.FESpace();
   const int size = nodes.Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes.GetData();
   double *temp = new double[size];
   for (int k=0; k<size; k++) temp[k]=0.0;
   for (int k=0,d = 0; d < ndofs; d++)
      for (int v = 0; v < vdim; v++)      
         temp[k++] = data[d+v*ndofs];
   for (int i=0; i<size; i++)
      data[i] = temp[i];
   delete [] temp;
   pop();
}

// ***************************************************************************
void qGeometry::ReorderByNodes(GridFunction& nodes)
{
   push();
   const mfem::FiniteElementSpace *fes=nodes.FESpace();
   const int size = nodes.Size();
   const int vdim = fes->GetVDim();
   const int ndofs = fes->GetNDofs();
   double *data = nodes.GetData();
   double *temp = new double[size];
   for (int k=0; k<size; k++) temp[k]=0.0;
   for (int k=0, j=0; j < ndofs; j++)
      for (int i=0; i < vdim; i++)
         temp[j+i*ndofs] = data[k++];
   for (int i = 0; i < size; i++)
      data[i] = temp[i];
   delete [] temp;
   pop();
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
