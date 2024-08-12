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

#ifndef MFEM_LAGHOS_REMESH
#define MFEM_LAGHOS_REMESH

#include "mfem.hpp"

namespace mfem
{

namespace hydrodynamics
{

void OptimizeMesh(ParGridFunction &coord_x_in,
                  AnalyticCompositeSurface &surfaces,
                  const IntegrationRule &ir, const IntegrationRule &ir_bdr,
                  ParGridFunction &coord_x_out);

// x = [1.0 + a sin(c pi) + b] t.
// y = 1 + a sin(c pi t) + b t.
// The distance is the error in y.
class Curve_Sine_Top : public Analytic2DCurve
{
private:
   const double a, b, c, x_scale;

public:
   Curve_Sine_Top(const Array<int> &marker, double a_, double b_, double c_)
       : Analytic2DCurve(marker),
       a(a_), b(b_), c(c_), x_scale(1.0 + a * sin(c * M_PI) + b) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = x / x_scale;
      dist = y - (1.0 + a * sin(c * M_PI * t) + b * t);
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = x_scale * t;
      y = dist + 1.0 + a * sin(c * M_PI * t) + b * t;
   }

   virtual double dx_dt(double t) const override
   { return x_scale; }
   virtual double dy_dt(double t) const override
   { return a * c * M_PI * cos(c * M_PI * t) + b; }

   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * t); }
};

// x = 1 + a sin(s pi t) + b t.
// y = [1.0 + a sin(s pi) + b] t.
// The distance is the error in x.
class Curve_Sine_Right : public Analytic2DCurve
{
private:
   const double a, b, c, y_scale;

public:
   Curve_Sine_Right(const Array<int> &marker, double a_, double b_, double c_)
       : Analytic2DCurve(marker),
       a(a_), b(b_), c(c_), y_scale(1.0 + a * sin(c * M_PI) + b) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = y / y_scale;
      dist = x - (1.0 + a * sin(c * M_PI * t) + b * t);
   }
   void xy_of_t(double t,  double dist, double &x, double &y) const override
   {
      x = dist + 1.0 + a * sin(c * M_PI * t) + b * t;
      y = y_scale * t;
   }

   virtual double dx_dt(double t) const override
   { return 0.2 * c * M_PI * cos(c * M_PI * t) + b; }
   virtual double dy_dt(double t) const override
   { return y_scale; }


   virtual double dx_dtdt(double t) const override
   { return -a * c * c * M_PI * M_PI * sin(c * M_PI * t); }
   virtual double dy_dtdt(double t) const override
   { return 0.0; }
};

// x = 0.
// y = t.
// The distance is the error in x.
class Line_Left : public Analytic2DCurve
{
public:
   Line_Left(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = y;
      dist = x - 0.0;
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = dist + 0.0;
      y = t;
   }

   virtual double dx_dt(double t) const override { return 0.0; }
   virtual double dy_dt(double t) const override { return 1.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = t.
// y = 0.
// The distance is the error in y.
class Line_Bottom : public Analytic2DCurve
{
public:
   Line_Bottom(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = x;
      dist = y - 0.0;
   }
   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      x = t;
      y = dist + 0.0;
   }

   virtual double dx_dt(double t) const override { return 1.0; }
   virtual double dy_dt(double t) const override { return 0.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// t in (-pi, pi] is the angle. The radius r is given. Center is at (0, 0).
// x = cos(t).
// y = sin(t).
// The distance is the error in radius.
class Circle : public Analytic2DCurve
{
private:
   const double r;

public:
   Circle(const Array<int> &marker, double rad)
    : Analytic2DCurve(marker), r(rad) { }

   void t_of_xy(double x, double y, double &dist, double &t) const override
   {
      t    = atan2(y, x);
      dist = sqrt(x*x + y*y) - r;
   }

   void xy_of_t(double t, double dist, double &x, double &y) const override
   {
      const double rad = dist + r;
      x = rad * cos(t);
      y = rad * sin(t);
   }

   virtual double dx_dt(double t) const override { return - r * sin(t); }
   virtual double dy_dt(double t) const override { return   r * cos(t); }

   virtual double dx_dtdt(double t) const override { return - r * r * cos(t); }
   virtual double dy_dtdt(double t) const override { return - r * r * sin(t); }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_REMESH
