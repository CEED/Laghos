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
                  ParGridFunction &coord_x_out);

// x = t, y = 0.
class Line_Bottom : public Analytic2DCurve
{
   public:
   Line_Bottom(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = t;
      y = 0.0;
   }
   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = x;
   }

   virtual double dx_dt(double t) const override { return 1.0; }
   virtual double dy_dt(double t) const override { return 0.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 0, y = t.
class Line_Left : public Analytic2DCurve
{
   public:
   Line_Left(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = 0.0;
      y = t;
   }
   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = y;
   }

   virtual double dx_dt(double t) const override { return 0.0; }
   virtual double dy_dt(double t) const override { return 1.0; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 1.5 t, y = 1 + 0.5 t
class Line_Top : public Analytic2DCurve
{
   public:
   Line_Top(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = x / 1.5;
   }
   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = 1.5 * t; y = 1.0 + 0.5 * t;
   }

   virtual double dx_dt(double t) const override { return 1.5; }
   virtual double dy_dt(double t) const override { return 0.5; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = 1 + 0.5 t, y = 1.5 t
class Line_Right : public Analytic2DCurve
{
   public:
   Line_Right(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = y / 1.5;
   }
   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = 1.0 + 0.5 * t; y = 1.5 * t;
   }

   virtual double dx_dt(double t) const override { return 0.5; }
   virtual double dy_dt(double t) const override { return 1.5; }
   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override { return 0.0; }
};

// x = [1.5 + 0.2 sin(s pi)] t
// y = 1 + 0.2 sin(s pi t) + 0.5 t
class Curve_Sine_Top : public Analytic2DCurve
{
   private:
   const double s_scale = 1.3;
   const double x_scale = 1.0 + 0.2 * sin(s_scale * M_PI) + 0.5;

   public:
   Curve_Sine_Top(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = x / x_scale;
   }
   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = x_scale * t;
      y = 1.0 + 0.2 * sin(s_scale * M_PI * t) + 0.5 * t;
   }

   virtual double dx_dt(double t) const override
   { return x_scale; }
   virtual double dy_dt(double t) const override
   { return 0.2 * s_scale * M_PI * cos(s_scale * M_PI * t) + 0.5; }

   virtual double dx_dtdt(double t) const override { return 0.0; }
   virtual double dy_dtdt(double t) const override
   { return -0.2 * s_scale * s_scale * M_PI * M_PI * sin(s_scale * M_PI * t); }
};

// x = 1 + 0.2 sin(s pi t) + 0.5 t
// y = [1.5 + 0.2 sin(s pi)] t
class Curve_Sine_Right : public Analytic2DCurve
{
   private:
   const double s_scale = 1.3;
   const double y_scale = 1.0 + 0.2 * sin(s_scale * M_PI) + 0.5;

   public:
   Curve_Sine_Right(const Array<int> &marker) : Analytic2DCurve(marker) { }

   void t_of_xy(double x, double y, const Vector &dist, double &t) const override
   {
      t = y / y_scale;
   }
   void xy_of_t(double t, const Vector &dist, double &x, double &y) const override
   {
      x = 1.0 + 0.2 * sin(s_scale * M_PI * t) + 0.5 * t;
      y = y_scale * t;
   }

   virtual double dx_dt(double t) const override
   { return 0.2 * s_scale * M_PI * cos(s_scale * M_PI * t) + 0.5; }
   virtual double dy_dt(double t) const override
   { return y_scale; }


   virtual double dx_dtdt(double t) const override
   { return -0.2 * s_scale * s_scale * M_PI * M_PI * sin(s_scale * M_PI * t); }
   virtual double dy_dtdt(double t) const override
   { return 0.0; }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_REMESH
