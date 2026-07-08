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
                  double remesh_dist, ParGridFunction &coord_x_out, bool vis);

double EvaluateBoundaryDistance(ParGridFunction &coord_x,
                                const IntegrationRule &ir_bdr,
                                const AnalyticCompositeSurface &surfaces,
                                const Array<int> &be_to_surface,
                                bool visualize);

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

class AxisAlignedEdge : public Analytic3DCurve
{
private:
   const int axis;
   const double fixed_1, fixed_2;

public:
   AxisAlignedEdge(const Array<bool> &marker, int axis_,
                   double fixed_1_, double fixed_2_)
      : Analytic3DCurve(marker),
        axis(axis_), fixed_1(fixed_1_), fixed_2(fixed_2_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      if (axis == 0)
      {
         t = x;
         dist1 = y - fixed_1;
         dist2 = z - fixed_2;
      }
      else if (axis == 1)
      {
         t = y;
         dist1 = x - fixed_1;
         dist2 = z - fixed_2;
      }
      else
      {
         t = z;
         dist1 = x - fixed_1;
         dist2 = y - fixed_2;
      }
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      if (axis == 0)
      {
         x = t;
         y = fixed_1 + dist1;
         z = fixed_2 + dist2;
      }
      else if (axis == 1)
      {
         x = fixed_1 + dist1;
         y = t;
         z = fixed_2 + dist2;
      }
      else
      {
         x = fixed_1 + dist1;
         y = fixed_2 + dist2;
         z = t;
      }
   }

   double dx_dt(double) const override { return (axis == 0) ? 1.0 : 0.0; }
   double dy_dt(double) const override { return (axis == 1) ? 1.0 : 0.0; }
   double dz_dt(double) const override { return (axis == 2) ? 1.0 : 0.0; }

   double dx_dtdt(double) const override { return 0.0; }
   double dy_dtdt(double) const override { return 0.0; }
   double dz_dtdt(double) const override { return 0.0; }
};

class AxisAlignedPlane : public Analytic3DSurface
{
private:
   const int normal_axis;
   const double fixed_value;

public:
   AxisAlignedPlane(const Array<bool> &marker, int normal_axis_,
                    double fixed_value_)
      : Analytic3DSurface(marker),
        normal_axis(normal_axis_), fixed_value(fixed_value_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      if (normal_axis == 0)
      {
         dist = x - fixed_value;
         u = y;
         v = z;
      }
      else if (normal_axis == 1)
      {
         dist = y - fixed_value;
         u = x;
         v = z;
      }
      else
      {
         dist = z - fixed_value;
         u = x;
         v = y;
      }
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      if (normal_axis == 0)
      {
         x = fixed_value + dist;
         y = u;
         z = v;
      }
      else if (normal_axis == 1)
      {
         x = u;
         y = fixed_value + dist;
         z = v;
      }
      else
      {
         x = u;
         y = v;
         z = fixed_value + dist;
      }
   }

   double dx_du(double, double) const override
   { return (normal_axis == 1 || normal_axis == 2) ? 1.0 : 0.0; }
   double dy_du(double, double) const override
   { return (normal_axis == 0) ? 1.0 : 0.0; }
   double dz_du(double, double) const override { return 0.0; }
   double dx_dv(double, double) const override { return 0.0; }
   double dy_dv(double, double) const override
   { return (normal_axis == 2) ? 1.0 : 0.0; }
   double dz_dv(double, double) const override
   { return (normal_axis == 0 || normal_axis == 1) ? 1.0 : 0.0; }

   double dx_dudu(double, double) const override { return 0.0; }
   double dy_dudu(double, double) const override { return 0.0; }
   double dz_dudu(double, double) const override { return 0.0; }
   double dx_dudv(double, double) const override { return 0.0; }
   double dy_dudv(double, double) const override { return 0.0; }
   double dz_dudv(double, double) const override { return 0.0; }
   double dx_dvdv(double, double) const override { return 0.0; }
   double dy_dvdv(double, double) const override { return 0.0; }
   double dz_dvdv(double, double) const override { return 0.0; }
};

class TorusSurface : public Analytic3DSurface
{
private:
   const double major_radius;
   const double minor_radius;

   double tube_radius(double v) const
   {
      return major_radius + minor_radius * cos(v);
   }

public:
   TorusSurface(const Array<bool> &marker,
                double major_radius_, double minor_radius_)
      : Analytic3DSurface(marker),
        major_radius(major_radius_), minor_radius(minor_radius_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      const double radial_xy = sqrt(x * x + y * y);
      const double radial_tube = radial_xy - major_radius;
      u = atan2(y, x);
      v = atan2(z, radial_tube);
      dist = sqrt(radial_tube * radial_tube + z * z) - minor_radius;
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      const double radius = tube_radius(v) + dist * cos(v);
      x = radius * cos(u);
      y = radius * sin(u);
      z = (minor_radius + dist) * sin(v);
   }

   double dx_du(double u, double v) const override
   { return -tube_radius(v) * sin(u); }
   double dy_du(double u, double v) const override
   { return tube_radius(v) * cos(u); }
   double dz_du(double, double) const override
   { return 0.0; }
   double dx_dv(double u, double v) const override
   { return -minor_radius * sin(v) * cos(u); }
   double dy_dv(double u, double v) const override
   { return -minor_radius * sin(v) * sin(u); }
   double dz_dv(double, double v) const override
   { return minor_radius * cos(v); }

   double dx_dudu(double u, double v) const override
   { return -tube_radius(v) * cos(u); }
   double dy_dudu(double u, double v) const override
   { return -tube_radius(v) * sin(u); }
   double dz_dudu(double, double) const override
   { return 0.0; }
   double dx_dudv(double u, double v) const override
   { return minor_radius * sin(v) * sin(u); }
   double dy_dudv(double u, double v) const override
   { return -minor_radius * sin(v) * cos(u); }
   double dz_dudv(double, double) const override
   { return 0.0; }
   double dx_dvdv(double u, double v) const override
   { return -minor_radius * cos(v) * cos(u); }
   double dy_dvdv(double u, double v) const override
   { return -minor_radius * cos(v) * sin(u); }
   double dz_dvdv(double, double v) const override
   { return -minor_radius * sin(v); }
};

class CubeCornerEdge : public Analytic3DCurve
{
private:
   const int axis;
   const double b;

   double coord(double t) const { return 1.0 + b * t; }

public:
   CubeCornerEdge(const Array<bool> &marker, int axis_, double b_)
      : Analytic3DCurve(marker), axis(axis_), b(b_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      const double val = (axis == 0) ? x : (axis == 1) ? y : z;
      t = val / (1.0 + b);
      dist1 = ((axis == 0) ? y : x) - coord(t);
      dist2 = ((axis == 2) ? y : z) - coord(t);
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      if (axis == 0)
      {
         x = (1.0 + b) * t;
         y = coord(t) + dist1;
         z = coord(t) + dist2;
      }
      else if (axis == 1)
      {
         x = coord(t) + dist1;
         y = (1.0 + b) * t;
         z = coord(t) + dist2;
      }
      else
      {
         x = coord(t) + dist1;
         y = coord(t) + dist2;
         z = (1.0 + b) * t;
      }
   }

   double dx_dt(double) const override { return (axis == 0) ? 1.0 + b : b; }
   double dy_dt(double) const override { return (axis == 1) ? 1.0 + b : b; }
   double dz_dt(double) const override { return (axis == 2) ? 1.0 + b : b; }

   double dx_dtdt(double) const override { return 0.0; }
   double dy_dtdt(double) const override { return 0.0; }
   double dz_dtdt(double) const override { return 0.0; }
};

class CubeCornerFace : public Analytic3DSurface
{
private:
   const int normal_axis;
   const double b;

   double warped(double u, double v) const { return 1.0 + b * u * v; }

public:
   CubeCornerFace(const Array<bool> &marker, int normal_axis_, double b_)
      : Analytic3DSurface(marker), normal_axis(normal_axis_), b(b_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      if (normal_axis == 0)
      {
         u = y;
         v = z;
      }
      else if (normal_axis == 1)
      {
         u = x;
         v = z;
      }
      else
      {
         u = x;
         v = y;
      }

      for (int it = 0; it < 12; it++)
      {
         double r1, r2, j11, j12, j21, j22;
         if (normal_axis == 0)
         {
            r1 = y_of_uv(u, v) - y;
            r2 = z_of_uv(u, v) - z;
            j11 = dy_du(u, v);
            j12 = dy_dv(u, v);
            j21 = dz_du(u, v);
            j22 = dz_dv(u, v);
         }
         else if (normal_axis == 1)
         {
            r1 = x_of_uv(u, v) - x;
            r2 = z_of_uv(u, v) - z;
            j11 = dx_du(u, v);
            j12 = dx_dv(u, v);
            j21 = dz_du(u, v);
            j22 = dz_dv(u, v);
         }
         else
         {
            r1 = x_of_uv(u, v) - x;
            r2 = y_of_uv(u, v) - y;
            j11 = dx_du(u, v);
            j12 = dx_dv(u, v);
            j21 = dy_du(u, v);
            j22 = dy_dv(u, v);
         }
         const double det = j11 * j22 - j12 * j21;
         if (fabs(det) < 1e-12) { break; }
         const double du = (-r1 * j22 + r2 * j12) / det;
         const double dv = (-j11 * r2 + j21 * r1) / det;
         u += du;
         v += dv;
         if (std::max(fabs(du), fabs(dv)) < 1e-12) { break; }
      }

      if (normal_axis == 0) { dist = x - x_of_uv(u, v); }
      else if (normal_axis == 1) { dist = y - y_of_uv(u, v); }
      else { dist = z - z_of_uv(u, v); }
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      x = x_of_uv(u, v);
      y = y_of_uv(u, v);
      z = z_of_uv(u, v);
      if (normal_axis == 0) { x += dist; }
      else if (normal_axis == 1) { y += dist; }
      else { z += dist; }
   }

   double dx_du(double, double v) const override
   { return (normal_axis == 0) ? b * v : 1.0 + b * v; }
   double dy_du(double, double v) const override
   { return (normal_axis == 0) ? 1.0 + b * v : b * v; }
   double dz_du(double, double v) const override
   { return b * v; }
   double dx_dv(double u, double) const override
   { return b * u; }
   double dy_dv(double u, double) const override
   { return (normal_axis == 2) ? 1.0 + b * u : b * u; }
   double dz_dv(double u, double) const override
   { return (normal_axis == 2) ? b * u : 1.0 + b * u; }

   double dx_dudu(double, double) const override { return 0.0; }
   double dy_dudu(double, double) const override { return 0.0; }
   double dz_dudu(double, double) const override { return 0.0; }
   double dx_dudv(double, double) const override { return b; }
   double dy_dudv(double, double) const override { return b; }
   double dz_dudv(double, double) const override { return b; }
   double dx_dvdv(double, double) const override { return 0.0; }
   double dy_dvdv(double, double) const override { return 0.0; }
   double dz_dvdv(double, double) const override { return 0.0; }

private:
   double x_of_uv(double u, double v) const
   { return (normal_axis == 0) ? warped(u, v) : u + b * u * v; }
   double y_of_uv(double u, double v) const
   {
      return (normal_axis == 1) ? warped(u, v) :
             (normal_axis == 0) ? u + b * u * v : v + b * u * v;
   }
   double z_of_uv(double u, double v) const
   { return (normal_axis == 2) ? warped(u, v) : v + b * u * v; }
};

inline void SineCubeTransform(double a, double b, double c,
                              double r, double s, double q,
                              double &x, double &y, double &z)
{
   const double h = 0.5 * M_PI, cp = c * M_PI;
   x = r + a * sin(h * r) * sin(cp * s) * sin(cp * q) + b * r * s * q;
   y = s + a * sin(cp * r) * sin(h * s) * sin(cp * q) + b * r * s * q;
   z = q + a * sin(cp * r) * sin(cp * s) * sin(h * q) + b * r * s * q;
}

inline double SineCubeDerivative(double a, double b, double c, int comp, int d,
                                 double r, double s, double q)
{
   const double h = 0.5 * M_PI, cp = c * M_PI;
   if (comp == 0)
   {
      if (d == 0)
      { return 1.0 + a * h * cos(h * r) * sin(cp * s) * sin(cp * q) + b * s * q; }
      if (d == 1)
      { return a * cp * sin(h * r) * cos(cp * s) * sin(cp * q) + b * r * q; }
      return a * cp * sin(h * r) * sin(cp * s) * cos(cp * q) + b * r * s;
   }
   if (comp == 1)
   {
      if (d == 0)
      { return a * cp * cos(cp * r) * sin(h * s) * sin(cp * q) + b * s * q; }
      if (d == 1)
      { return 1.0 + a * h * sin(cp * r) * cos(h * s) * sin(cp * q) + b * r * q; }
      return a * cp * sin(cp * r) * sin(h * s) * cos(cp * q) + b * r * s;
   }

   if (d == 0)
   { return a * cp * cos(cp * r) * sin(cp * s) * sin(h * q) + b * s * q; }
   if (d == 1)
   { return a * cp * sin(cp * r) * cos(cp * s) * sin(h * q) + b * r * q; }
   return 1.0 + a * h * sin(cp * r) * sin(cp * s) * cos(h * q) + b * r * s;
}

inline double SineCubeSecondDerivative(double a, double b, double c,
                                       int comp, int d1, int d2,
                                       double r, double s, double q)
{
   const double h = 0.5 * M_PI, cp = c * M_PI;
   if (d1 > d2) { std::swap(d1, d2); }

   if (comp == 0)
   {
      const double base = a * sin(h * r) * sin(cp * s) * sin(cp * q);
      if (d1 == 0 && d2 == 0) { return -h * h * base; }
      if (d1 == 1 && d2 == 1) { return -cp * cp * base; }
      if (d1 == 2 && d2 == 2) { return -cp * cp * base; }
      if (d1 == 0 && d2 == 1)
      { return a * h * cp * cos(h * r) * cos(cp * s) * sin(cp * q) + b * q; }
      if (d1 == 0 && d2 == 2)
      { return a * h * cp * cos(h * r) * sin(cp * s) * cos(cp * q) + b * s; }
      return a * cp * cp * sin(h * r) * cos(cp * s) * cos(cp * q) + b * r;
   }

   if (comp == 1)
   {
      const double base = a * sin(cp * r) * sin(h * s) * sin(cp * q);
      if (d1 == 0 && d2 == 0) { return -cp * cp * base; }
      if (d1 == 1 && d2 == 1) { return -h * h * base; }
      if (d1 == 2 && d2 == 2) { return -cp * cp * base; }
      if (d1 == 0 && d2 == 1)
      { return a * cp * h * cos(cp * r) * cos(h * s) * sin(cp * q) + b * q; }
      if (d1 == 0 && d2 == 2)
      { return a * cp * cp * cos(cp * r) * sin(h * s) * cos(cp * q) + b * s; }
      return a * h * cp * sin(cp * r) * cos(h * s) * cos(cp * q) + b * r;
   }

   const double base = a * sin(cp * r) * sin(cp * s) * sin(h * q);
   if (d1 == 0 && d2 == 0) { return -cp * cp * base; }
   if (d1 == 1 && d2 == 1) { return -cp * cp * base; }
   if (d1 == 2 && d2 == 2) { return -h * h * base; }
   if (d1 == 0 && d2 == 1)
   { return a * cp * cp * cos(cp * r) * cos(cp * s) * sin(h * q) + b * q; }
   if (d1 == 0 && d2 == 2)
   { return a * cp * h * cos(cp * r) * sin(cp * s) * cos(h * q) + b * s; }
   return a * cp * h * sin(cp * r) * cos(cp * s) * cos(h * q) + b * r;
}

class SineCubeEdge : public Analytic3DCurve
{
private:
   const int axis;
   const double a, b, c;

public:
   SineCubeEdge(const Array<bool> &marker, int axis_,
                double a_, double b_, double c_)
      : Analytic3DCurve(marker), axis(axis_), a(a_), b(b_), c(c_) { }

   void t_of_xyz(double x, double y, double z,
                 double &dist1, double &dist2, double &t) const override
   {
      const double val = (axis == 0) ? x : (axis == 1) ? y : z;
      t = val;
      for (int it = 0; it < 12; it++)
      {
         double xt, yt, zt;
         xyz_on_edge(t, xt, yt, zt);
         const double f = ((axis == 0) ? xt : (axis == 1) ? yt : zt) - val;
         const double df = (axis == 0) ? dx_dt(t) :
                           (axis == 1) ? dy_dt(t) : dz_dt(t);
         if (fabs(df) < 1e-12) { break; }
         const double dt = -f / df;
         t += dt;
         if (fabs(dt) < 1e-12) { break; }
      }

      double xt, yt, zt;
      xyz_on_edge(t, xt, yt, zt);
      dist1 = ((axis == 0) ? y : x) - ((axis == 0) ? yt : xt);
      dist2 = ((axis == 2) ? y : z) - ((axis == 2) ? yt : zt);
   }

   void xyz_of_t(double t, double dist1, double dist2,
                 double &x, double &y, double &z) const override
   {
      xyz_on_edge(t, x, y, z);
      if (axis == 0)
      {
         y += dist1;
         z += dist2;
      }
      else if (axis == 1)
      {
         x += dist1;
         z += dist2;
      }
      else
      {
         x += dist1;
         y += dist2;
      }
   }

   double dx_dt(double t) const override
   {
      const double r = (axis == 0) ? t : 1.0;
      const double s = (axis == 1) ? t : 1.0;
      const double q = (axis == 2) ? t : 1.0;
      return SineCubeDerivative(a, b, c, 0, axis, r, s, q);
   }
   double dy_dt(double t) const override
   {
      const double r = (axis == 0) ? t : 1.0;
      const double s = (axis == 1) ? t : 1.0;
      const double q = (axis == 2) ? t : 1.0;
      return SineCubeDerivative(a, b, c, 1, axis, r, s, q);
   }
   double dz_dt(double t) const override
   {
      const double r = (axis == 0) ? t : 1.0;
      const double s = (axis == 1) ? t : 1.0;
      const double q = (axis == 2) ? t : 1.0;
      return SineCubeDerivative(a, b, c, 2, axis, r, s, q);
   }

   double dx_dtdt(double t) const override { return second(0, axis, axis, t); }
   double dy_dtdt(double t) const override { return second(1, axis, axis, t); }
   double dz_dtdt(double t) const override { return second(2, axis, axis, t); }

private:
   void xyz_on_edge(double t, double &x, double &y, double &z) const
   {
      SineCubeTransform(a, b, c,
                        (axis == 0) ? t : 1.0,
                        (axis == 1) ? t : 1.0,
                        (axis == 2) ? t : 1.0, x, y, z);
   }

   double second(int comp, int d1, int d2, double t) const
   {
      return SineCubeSecondDerivative(a, b, c, comp, d1, d2,
                                      (axis == 0) ? t : 1.0,
                                      (axis == 1) ? t : 1.0,
                                      (axis == 2) ? t : 1.0);
   }
};

class SineCubeFace : public Analytic3DSurface
{
private:
   const int normal_axis;
   const double a, b, c;

public:
   SineCubeFace(const Array<bool> &marker, int normal_axis_,
                double a_, double b_, double c_)
      : Analytic3DSurface(marker), normal_axis(normal_axis_),
        a(a_), b(b_), c(c_) { }

   void uv_of_xyz(double x, double y, double z,
                  double &dist, double &u, double &v) const override
   {
      if (normal_axis == 0) { u = y; v = z; }
      else if (normal_axis == 1) { u = x; v = z; }
      else { u = x; v = y; }

      for (int it = 0; it < 12; it++)
      {
         double r1, r2, j11, j12, j21, j22;
         if (normal_axis == 0)
         {
            r1 = y_of_uv(u, v) - y;
            r2 = z_of_uv(u, v) - z;
            j11 = dy_du(u, v);
            j12 = dy_dv(u, v);
            j21 = dz_du(u, v);
            j22 = dz_dv(u, v);
         }
         else if (normal_axis == 1)
         {
            r1 = x_of_uv(u, v) - x;
            r2 = z_of_uv(u, v) - z;
            j11 = dx_du(u, v);
            j12 = dx_dv(u, v);
            j21 = dz_du(u, v);
            j22 = dz_dv(u, v);
         }
         else
         {
            r1 = x_of_uv(u, v) - x;
            r2 = y_of_uv(u, v) - y;
            j11 = dx_du(u, v);
            j12 = dx_dv(u, v);
            j21 = dy_du(u, v);
            j22 = dy_dv(u, v);
         }
         const double det = j11 * j22 - j12 * j21;
         if (fabs(det) < 1e-12) { break; }
         const double du = (-r1 * j22 + r2 * j12) / det;
         const double dv = (-j11 * r2 + j21 * r1) / det;
         u += du;
         v += dv;
         if (std::max(fabs(du), fabs(dv)) < 1e-12) { break; }
      }

      if (normal_axis == 0) { dist = x - x_of_uv(u, v); }
      else if (normal_axis == 1) { dist = y - y_of_uv(u, v); }
      else { dist = z - z_of_uv(u, v); }
   }

   void xyz_of_uv(double u, double v, double dist,
                  double &x, double &y, double &z) const override
   {
      xyz_on_face(u, v, x, y, z);
      if (normal_axis == 0) { x += dist; }
      else if (normal_axis == 1) { y += dist; }
      else { z += dist; }
   }

   double dx_du(double u, double v) const override { return partial(0, 0, u, v); }
   double dy_du(double u, double v) const override { return partial(1, 0, u, v); }
   double dz_du(double u, double v) const override { return partial(2, 0, u, v); }
   double dx_dv(double u, double v) const override { return partial(0, 1, u, v); }
   double dy_dv(double u, double v) const override { return partial(1, 1, u, v); }
   double dz_dv(double u, double v) const override { return partial(2, 1, u, v); }

   double dx_dudu(double u, double v) const override
   { return second_partial(0, 0, 0, u, v); }
   double dy_dudu(double u, double v) const override
   { return second_partial(1, 0, 0, u, v); }
   double dz_dudu(double u, double v) const override
   { return second_partial(2, 0, 0, u, v); }
   double dx_dudv(double u, double v) const override
   { return second_partial(0, 0, 1, u, v); }
   double dy_dudv(double u, double v) const override
   { return second_partial(1, 0, 1, u, v); }
   double dz_dudv(double u, double v) const override
   { return second_partial(2, 0, 1, u, v); }
   double dx_dvdv(double u, double v) const override
   { return second_partial(0, 1, 1, u, v); }
   double dy_dvdv(double u, double v) const override
   { return second_partial(1, 1, 1, u, v); }
   double dz_dvdv(double u, double v) const override
   { return second_partial(2, 1, 1, u, v); }

private:
   int u_axis() const { return (normal_axis == 0) ? 1 : 0; }
   int v_axis() const { return (normal_axis == 2) ? 1 : 2; }

   void ref_coords(double u, double v, double &r, double &s, double &q) const
   {
      r = (normal_axis == 0) ? 1.0 : u;
      s = (normal_axis == 1) ? 1.0 : (normal_axis == 0) ? u : v;
      q = (normal_axis == 2) ? 1.0 : v;
   }

   void xyz_on_face(double u, double v,
                    double &x, double &y, double &z) const
   {
      double r, s, q;
      ref_coords(u, v, r, s, q);
      transform(r, s, q, x, y, z);
   }

   double x_of_uv(double u, double v) const
   { double x, y, z; xyz_on_face(u, v, x, y, z); return x; }
   double y_of_uv(double u, double v) const
   { double x, y, z; xyz_on_face(u, v, x, y, z); return y; }
   double z_of_uv(double u, double v) const
   { double x, y, z; xyz_on_face(u, v, x, y, z); return z; }

   double partial(int comp, int uv, double u, double v) const
   {
      double r, s, q;
      ref_coords(u, v, r, s, q);
      return SineCubeDerivative(a, b, c, comp,
                                (uv == 0) ? u_axis() : v_axis(), r, s, q);
   }

   double second_partial(int comp, int uv1, int uv2,
                         double u, double v) const
   {
      double r, s, q;
      ref_coords(u, v, r, s, q);
      return SineCubeSecondDerivative(a, b, c, comp,
                                      (uv1 == 0) ? u_axis() : v_axis(),
                                      (uv2 == 0) ? u_axis() : v_axis(),
                                      r, s, q);
   }

   void transform(double r, double s, double q,
                  double &x, double &y, double &z) const
   {
      SineCubeTransform(a, b, c, r, s, q, x, y, z);
   }
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_REMESH
