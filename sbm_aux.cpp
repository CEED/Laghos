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

#include "sbm_aux.hpp"

using namespace std;
using namespace mfem;

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
double relativePosition(const Vector &x, const int type)
{
   if (type == 1) // circle of radius 0.2 - centered at 0.5, 0.5
   {
     Vector center(2);
     center(0) = 0.5;
     center(1) = 0.5;
     double radiusOfPt = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
     const double radius = 0.2;
     return radiusOfPt - radius; // positive is the domain
   }
   else if (type == 2) // circle of radius 0.2 - centered at 0.5, 0.5
   {
     double slope = 0.0;
     double yIntercept = 0.6;
     double ptOnLine = slope * x(0) + yIntercept;
     return ptOnLine-x(1); // positive is the domain
   }
   else if (type == 3) // circle of radius 0.2 - centered at 0.5, 0.5
   {
     double xVertLine = 0.3;
     return xVertLine - x(0); // positive is the domain
   }

   else
     {
      MFEM_ABORT(" Function type not implement yet.");
   }
   return 0.;
}

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Circle_Dist(const Vector &x, Vector &D){
  double radius = 0.2;
  Vector center(2);
  center(0) = 0.5;
  center(1) = 0.5;
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  D(0) = distX;
  D(1) = distY;
}

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Circle_Normal(const Vector &x, Vector &tN){
  double radius = 0.2;
  Vector center(2);
  center(0) = 0.5;
  center(1) = 0.5;
  
  double r = pow(pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0),0.5);
  double distX = ((x(0)-center(0))/r)*(radius-r);
  double distY = ((x(1)-center(1))/r)*(radius-r);
  double normD = sqrt(distX * distX + distY * distY);
  if (r < radius){
    tN(0) = -distX / normD;
    tN(1) = -distY / normD;
  }
  else if (r > radius){
    tN(0) = distX / normD;
    tN(1) = distY / normD;
  }
  else{
    tN(0) = (center(0) - x(0))/radius;
    tN(1) = (center(1) - x(1))/radius;
  }
}

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void Line_Dist(const Vector &x, Vector &D){
  double slope = 0.0;
  double yIntercept = 0.6;
  double ptOnLine = slope * x(0) + yIntercept;
  D(0) = 0.0;
  D(1) = ptOnLine - x(1);
}

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void Line_Normal(const Vector &x, Vector &tN){
  tN(0) = 0.0;
  tN(1) = 1.0;
}

// Unit normal of circle of radius 0.2 - centered at 0.5, 0.5
void VerticalLine_Normal(const Vector &x, Vector &tN){
  tN(0) = 1.0;
  tN(1) = 0.0;
}

// Distance to circle of radius 0.2 - centered at 0.5, 0.5 
void VerticalLine_Dist(const Vector &x, Vector &D){
  double xVertLine = 0.3;
  D(0) = xVertLine - x(0) ;
  D(1) = 0.0;
}

/// Analytic distance to the 0 level set.
void dist_value(const Vector &x, Vector &D, const int type)
{
   if (type == 1) {
     return Circle_Dist(x, D);
   }
   else if (type == 2) {
     return Line_Dist(x, D);
   }
   else if (type == 3) {
     return VerticalLine_Dist(x, D);
   }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
}

/// Analytic distance to the 0 level set. Positive value if the point is inside
/// the domain, and negative value if outside.
void normal_value(const Vector &x, Vector &tN, const int type)
{
   if (type == 1) {
     return Circle_Normal(x, tN);
   }
   else if (type == 2) {
     return Line_Normal(x, tN);
   }
   else if (type == 3) {
     return VerticalLine_Normal(x, tN);
   }
   else
   {
      MFEM_ABORT(" Function type not implement yet.");
   }
}
