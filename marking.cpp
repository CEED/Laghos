// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "marking.hpp"

namespace mfem
{

void ShiftedFaceMarker::MarkElements(const ParGridFunction &ls_func)
{
      MPI_Comm comm = pmesh.GetComm();
    int myid;
    MPI_Comm_rank(comm, &myid);

    const int max_elem_attr = (pmesh.attributes).Max();
    int activeCount = 0;
    int inactiveCount = 0;
    int cutCount = 0;

    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);

    // This tolerance is relevant for points that are exactly on the zero LS.
    const double eps = 1e-16;
    auto outside_of_domain = [&](double value)
      {
	if (include_cut_cell)
	  {
	    // Points on the zero LS are considered outside the domain.
	    return (value - eps < 0.0);
	  }
	else
	  {
	    // Points on the zero LS are considered inside the domain.
	    return (value + eps < 0.0);
	  }
      };
    ParFiniteElementSpace * ls_fes = ls_func.ParFESpace();
    Vector vals;
    // Check elements on the current MPI rank
    for (int i = 0; i < pmesh.GetNE(); i++)
      {
	const IntegrationRule &ir = ls_fes->GetFE(i)->GetNodes();
	ls_func.GetValues(i, ir, vals);
	int count = 0;
	for (int j = 0; j < ir.GetNPoints(); j++)
	  {
	    if (outside_of_domain(vals(j))) { count++; }
	  }

	if (count == ir.GetNPoints()) // completely outside
	  {     
	    inactiveCount++;
	    pmesh.SetAttribute(i, SBElementType::OUTSIDE);
	    mat_attr(i) = SBElementType::OUTSIDE;
	  }
	else if ((count > 0) && (count < ir.GetNPoints())) // partially outside
	  {

	    cutCount++;	
	    pmesh.SetAttribute(i, SBElementType::CUT);
	    mat_attr(i) = SBElementType::CUT;		
	    if (include_cut_cell){
	      Array<int> dofs;
	      pfes_sltn->GetElementVDofs(i, dofs);
	      for (int k = 0; k < dofs.Size(); k++)
		{
		  ess_inactive[dofs[k]] = 0;
		}
	    }
	  }
	else // inside
	  {
	    pmesh.SetAttribute(i, SBElementType::INSIDE);
	    mat_attr(i) = SBElementType::INSIDE;		
	    activeCount++;
	    Array<int> dofs;
	    pfes_sltn->GetElementVDofs(i, dofs);
	    for (int k = 0; k < dofs.Size(); k++)
	      {
		ess_inactive[dofs[k]] = 0;	       
	      }
	  }
      }
    mat_attr.ExchangeFaceNbrData(); 
    pmesh.ExchangeFaceNbrNodes();

    for (int f = 0; f < pmesh.GetNumFaces(); f++)
      {
	auto *ft = pmesh.GetFaceElementTransformations(f, 3);
	if (ft->Elem2No > 0) {
	  bool elem1_inside = (pmesh.GetAttribute(ft->Elem1No) == SBElementType::INSIDE);
	  bool elem1_cut = (pmesh.GetAttribute(ft->Elem1No) == SBElementType::CUT);
	  bool elem1_outside = (pmesh.GetAttribute(ft->Elem1No) == SBElementType::OUTSIDE);
  
	  bool elem2_inside = (pmesh.GetAttribute(ft->Elem2No) == SBElementType::INSIDE);
	  bool elem2_cut = (pmesh.GetAttribute(ft->Elem2No) == SBElementType::CUT);
	  bool elem2_outside = (pmesh.GetAttribute(ft->Elem2No) == SBElementType::OUTSIDE);
	  // ghost faces
	  // outer surrogate boundaries      
	  if ( elem1_cut && elem2_inside  ) {
	    pmesh.SetFaceAttribute(f, 21);	
	    Array<int> dofs;
	    pfes_sltn->GetFaceVDofs(f, dofs);
	    for (int k = 0; k < dofs.Size(); k++)
	      {
		surrogateNodes(dofs[k]) = 1;	       
	      }	  
	  }
	  else if (elem1_inside && elem2_cut) {
	    pmesh.SetFaceAttribute(f, 12);
	    Array<int> dofs;
	    pfes_sltn->GetFaceVDofs(f, dofs);
	    
	    for (int k = 0; k < dofs.Size(); k++)
	      {
		surrogateNodes(dofs[k]) = 1;	       
	      }	  	 
	  }
	  else if (elem1_inside && elem2_inside) {
	    pmesh.SetFaceAttribute(f, 33);
	  }
	}
      }
    pmesh.ExchangeFaceNbrNodes();
    //    surrogateNodes.ExchangeFaceNbrData();   
    //  std::cout << " owned face " << pmesh.GetNumFaces() << std::endl;
    // surrogateNodes.ExchangeFaceNbrData();
    const int c_vsize = pfes_sltn->GetVSize();
    for (int f = 0; f < pmesh.GetNSharedFaces(); f++)
      {
	auto *ftr = pmesh.GetSharedFaceTransformations(f, 3);
	int faceno = pmesh.GetSharedFace(f);
	const bool ghost_sface = (faceno >= pmesh.GetNumFaces());
	int Elem2NbrNo = ftr->Elem2No - pmesh.GetNE();
	auto *nbrftr = ls_fes->GetFaceNbrElementTransformation(Elem2NbrNo);
	int attr1 = pmesh.GetAttribute(ftr->Elem1No);
	IntegrationPoint sip; sip.Init(0);
	int attr2 = mat_attr.GetValue(*ftr->Elem2, sip);
	bool elem1_inside = (attr1 == SBElementType::INSIDE);
	bool elem1_cut = (attr1 == SBElementType::CUT);
	bool elem1_outside = (attr1 == SBElementType::OUTSIDE);
       
	bool elem2_inside = (attr2 == SBElementType::INSIDE);
	bool elem2_cut = (attr2 == SBElementType::CUT);
	bool elem2_outside = (attr2 == SBElementType::OUTSIDE);
	// outer surrogate boundaries
	if ( elem1_cut && elem2_inside ) {
	  pmesh.SetFaceAttribute(faceno, 21);
	  Array<int> dofs;
	  if (!ghost_sface){
	    pfes_sltn->GetFaceVDofs(faceno, dofs);	    
	  }
	  else{
	    pfes_sltn->GetFaceNbrFaceVDofs(faceno, dofs);
	    FiniteElementSpace::AdjustVDofs(dofs);
	    for (int j = 0; j < dofs.Size(); j++)
	      {
		dofs[j] += c_vsize;
	      }
	  }
	  for (int k = 0; k < dofs.Size(); k++)
	    {
	      surrogateNodes(dofs[k]) = 1;
	    }	  
	}
	else if (elem1_inside && elem2_cut){
	  pmesh.SetFaceAttribute(faceno, 12);
	  Array<int> dofs;
	  if (!ghost_sface){
	    pfes_sltn->GetFaceVDofs(faceno, dofs);	   
	  }
	  else{
	    pfes_sltn->GetFaceNbrFaceVDofs(faceno, dofs);
	    FiniteElementSpace::AdjustVDofs(dofs);
	    for (int j = 0; j < dofs.Size(); j++)
	      {
		dofs[j] += c_vsize;
	      }
	  }
	  for (int k = 0; k < dofs.Size(); k++)
	    {
	      surrogateNodes(dofs[k]) = 1;
	    }	  
	}
	else if (elem1_inside && elem2_inside){
	  pmesh.SetFaceAttribute(faceno, 33);
	}
      }
    
    surrogateNodes.ExchangeFaceNbrData();
    /*  GridFunctionCoefficient surrogateNodesCoef(&surrogateNodes);
    parallelSurrogateNodes.ProjectCoefficient(surrogateNodesCoef);
    parallelSurrogateNodes.ExchangeFaceNbrData();*/
    //  std::cout << " myid " << myid << std::endl;
    //    surrogateNodes.Print();
    
    // Check elements on the current MPI rank
    for (int i = 0; i < pmesh.GetNE(); i++)
      {
	const IntegrationRule &ir = ls_fes->GetFE(i)->GetNodes();
	surrogateNodes.GetValues(i, ir, vals);
	int count = 0;
	for (int j = 0; j < ir.GetNPoints(); j++)
	  {
	    if (vals(j) > 0.0 && pmesh.GetAttribute(i) != CUT) {
	      pmesh.SetAttribute(i, SBElementType::GHOST);
	      mat_attr(i) = SBElementType::GHOST;
	      break;
	    }
	  }
      }
    mat_attr.ExchangeFaceNbrData(); 
    pmesh.ExchangeFaceNbrNodes();
    
    for (int f = 0; f < pmesh.GetNumFaces(); f++)
      {
	auto *ft = pmesh.GetFaceElementTransformations(f, 3);
	if (ft->Elem2No > 0) {
	  bool elem1_ghost = (pmesh.GetAttribute(ft->Elem1No) == SBElementType::GHOST);
	  bool elem1_inside = (pmesh.GetAttribute(ft->Elem1No) == SBElementType::INSIDE);
	 
	  bool elem2_inside = (pmesh.GetAttribute(ft->Elem2No) == SBElementType::INSIDE); 
	  bool elem2_ghost = (pmesh.GetAttribute(ft->Elem2No) == SBElementType::GHOST);
	  // outer surrogate boundaries      
	  if ( (elem1_ghost && elem2_inside) || (elem1_inside && elem2_ghost) || (elem1_ghost && elem2_ghost)  ) {
	    pmesh.SetFaceAttribute(f, 77);
	  }
	}
      }
   
    for (int f = 0; f < pmesh.GetNSharedFaces(); f++)
      {
	auto *ftr = pmesh.GetSharedFaceTransformations(f, 3);
	int faceno = pmesh.GetSharedFace(f);
	int Elem2NbrNo = ftr->Elem2No - pmesh.GetNE();
	auto *nbrftr = ls_fes->GetFaceNbrElementTransformation(Elem2NbrNo);
	int attr1 = pmesh.GetAttribute(ftr->Elem1No);
	IntegrationPoint sip; sip.Init(0);
	int attr2 = mat_attr.GetValue(*ftr->Elem2, sip);
	bool elem1_inside = (attr1 == SBElementType::INSIDE);
	bool elem1_ghost = (attr1 == SBElementType::GHOST);
       
	bool elem2_inside = (attr2 == SBElementType::INSIDE);
	bool elem2_ghost = (attr2 == SBElementType::GHOST);
	if ( (elem1_ghost && elem2_inside) || (elem1_inside && elem2_ghost) || (elem1_ghost && elem2_ghost)  ) {
	  pmesh.SetFaceAttribute(faceno, 77);	
	}
      }
    
    pmesh.ExchangeFaceNbrNodes();

    initial_marking_done = true;
   std::cout << " active elemSta " << activeCount << " cut " << cutCount << " inacive " << inactiveCount <<  std::endl;
   // Synchronize
   for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] += 1; }
   pfes_sltn->Synchronize(ess_inactive);
   for (int i = 0; i < ess_inactive.Size() ; i++) { ess_inactive[i] -= 1; }
   pmesh.SetAttributes();
}

  Array<int>& ShiftedFaceMarker::GetEss_Vdofs(){
    return ess_inactive;
  }

}
