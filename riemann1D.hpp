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

#include "mfem.hpp"

namespace riemann1D
{

using namespace mfem;

double init(const double *params);
double rho(const double *xt);
double p(const double *xt);
double v(const double *xt);
double e(const double *xt);

class ExactEnergyCoefficient : public Coefficient
{
public:
   ExactEnergyCoefficient()
   {
      double params[8];
      params[0] = 1.0; params[3] = 0.1; // rho
      params[1] = 1.0; params[4] = 0.1; // p
      params[2] = params[5] = 0.0;      // u
      params[6] = 1.4;                  // gamma
      params[7] = 0.5;                  // x_center
      init(params);
   }

   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      Vector pos(3);
      T.Transform(ip, pos);

      double p[2]; p[0] = pos(0); p[1] = time;
      return e(p);
   }
};

const double rel_err = 1e-14;
const int max_iter   = 30;

double rl, pl, ul, rr, pr, ur, gamma_, alpha_, tau_, xi;
double cl, cr, uc, pc, clc, crc, rcl, rcr, uls, urs;
char l_w, r_w;

double rho(const double *xt)
{
   double x = xt[0], t = xt[1];

   x -= xi;

   if (l_w == 'r')
   {
      double xll = t * (ul - cl);
      if (x <= xll)
      {
         return rl;
      }
      double xlr = t * (uc - clc);
      if (x < xlr)
      {
         double u = ul + (uc - ul) * (x - xll) / (xlr - xll);
         return rl * pow(1 - alpha_ / cl * (u - ul), 1 / alpha_);
      }
   }
   else
   {
      if (x <= t * uls)
      {
         return rl;
      }
   }

   double xc = t * uc;
   if (x <= xc)
   {
      return rcl;
   }

   if (r_w == 'r')
   {
      double xrl = t * (uc + crc);
      if (x <= xrl)
      {
         return rcr;
      }
      double xrr = t * (ur + cr);
      if (x < xrr)
      {
         double u = ur + (uc - ur) * (x - xrr) / (xrl - xrr);
         return rr * pow(1 + alpha_ / cr * (u - ur), 1 / alpha_);
      }
   }
   else
   {
      if (x <= t * urs)
      {
         return rcr;
      }
   }

   return rr;
}

double p(const double *xt)
{
   double x = xt[0], t = xt[1];

   x -= xi;

   if (l_w == 'r')
   {
      double xll = t * (ul - cl);
      if (x <= xll)
      {
         return pl;
      }
      double xlr = t * (uc - clc);
      if (x < xlr)
      {
         double u = ul + (uc - ul) * (x - xll) / (xlr - xll);
         return pl * pow(1 - alpha_ / cl * (u - ul), 1 / tau_);
      }
   }
   else
   {
      if (x <= t * uls)
      {
         return pl;
      }
   }

   if (r_w == 'r')
   {
      double xrl = t * (uc + crc);
      if (x <= xrl)
      {
         return pc;
      }
      double xrr = t * (ur + cr);
      if (x < xrr)
      {
         double u = ur + (uc - ur) * (x - xrr) / (xrl - xrr);
         return pr * pow(1 + alpha_ / cr * (u - ur), 1 / tau_);
      }
   }
   else
   {
      if (x <= t * urs)
      {
         return pc;
      }
   }

   return pr;
}

double v(const double *xt)
{
   double x = xt[0], t = xt[1];

   x -= xi;

   if (l_w == 'r')
   {
      double xll = t * (ul - cl);
      if (x <= xll)
      {
         return ul;
      }
      double xlr = t * (uc - clc);
      if (x < xlr)
      {
         return ul + (uc - ul) * (x - xll) / (xlr - xll);
      }
   }
   else
   {
      if (x <= t * uls)
      {
         return ul;
      }
   }

   if (r_w == 'r')
   {
      double xrl = t * (uc + crc);
      if (x <= xrl)
      {
         return uc;
      }
      double xrr = t * (ur + cr);
      if (x < xrr)
      {
         return ur + (uc - ur) * (x - xrr) / (xrl - xrr);
      }
   }
   else
   {
      if (x <= t * urs)
      {
         return uc;
      }
   }

   return ur;
}

double e(const double *xt) { return p(xt) / ((gamma_ - 1) * rho(xt)); }

int exact_riemann_solver(

   // polytropic index
   const double gamma_,
   // left and right states: velocity, pressure, sound speed
   const double ul,
   const double pl,
   const double cl,
   const double ur,
   const double pr,
   const double cr,
   // relative error (of pressure) and maximum number of iterations
   const double rel_err,
   const int max_iter,

   // contact velocity, pressure, left and right sound speed
   double &uc,
   double &pc,
   double &clc,
   double &crc,
   // left and right wave type: 'r' or 's': rarefaction or shock
   char &l_w,
   char &r_w)
{
   double gm = gamma_ - 1, hgm = gm / 2, ghgm = gamma_ / hgm;
   double gp = gamma_ + 1, qgp = gp / 4;
   double rocl = gamma_ * pl / cl, rocr = gamma_ * pr / cr;
   double z, ult, urt, dul, dur, aa, plc, prc, plcp, prcp, wl, wr;

   // initial guess for contact velocity
   z   = pow(pl / pr, 1 / ghgm) * cr / cl;
   ult = ul + cl / hgm;
   urt = ur - cr / hgm;
   uc  = (ult * z + urt) / (1 + z);

   // solve the Riemann problem
   for (int it = 0; it < max_iter; it++)
   {
      dul = uc - ul;

      if (dul <= 0)  // leftward moving shock
      {
         aa   = qgp * dul / cl;
         wl   = aa - sqrt(1 + aa * aa);
         plc  = pl + rocl * dul * wl;
         plcp = 2 * rocl * pow(wl, 3) / (1 + wl * wl);
         clc  = cl * sqrt((gp + gm * plc / pl) / (gp + gm * pl / plc));
         l_w  = 's';
      }
      else  // left rarefaction wave
      {
         clc  = cl - hgm * dul;
         plc  = pl * pow(clc / cl, ghgm);
         plcp = -gamma_ * plc / clc;
         l_w  = 'r';
      }

      dur = uc - ur;

      if (dur >= 0)  // rightward moving shock
      {
         aa   = qgp * dur / cr;
         wr   = aa + sqrt(1 + aa * aa);
         prc  = pr + rocr * dur * wr;
         prcp = 2 * rocr * pow(wr, 3) / (1 + wr * wr);
         crc  = cr * sqrt((gp + gm * prc / pr) / (gp + gm * pr / prc));
         r_w  = 's';
      }
      else  // right rarefaction wave
      {
         crc  = cr + hgm * dur;
         prc  = pr * pow(crc / cr, ghgm);
         prcp = gamma_ * prc / crc;
         r_w  = 'r';
      }

      // convergence check

      if (fabs(1 - plc / prc) <= rel_err)
      {
         pc = plc;
         return 0;
      }

      uc = uc - (plc - prc) / (plcp - prcp);
   }

   // no convergence
   return 1;
}

double init(const double *params)
{
   rl = params[0];
   pl = params[1];
   ul = params[2];

   rr = params[3];
   pr = params[4];
   ur = params[5];

   gamma_ = params[6];

   xi = params[7];

   alpha_ = (gamma_ - 1) / 2;
   tau_   = alpha_ / gamma_;

   cl = sqrt(gamma_ * pl / rl);
   cr = sqrt(gamma_ * pr / rr);

   int err = exact_riemann_solver(gamma_,
                                  ul,
                                  pl,
                                  cl,
                                  ur,
                                  pr,
                                  cr,
                                  rel_err,
                                  max_iter,
                                  uc,
                                  pc,
                                  clc,
                                  crc,
                                  l_w,
                                  r_w);

   if (err)
   {
      mfem::mfem_error("The exact Riemann solver failed to converge!");
   }

   rcl = gamma_ * pc / (clc * clc);
   rcr = gamma_ * pc / (crc * crc);

   if (l_w == 's')
   {
      uls = (rl * ul - rcl * uc) / (rl - rcl);
   }
   if (r_w == 's')
   {
      urs = (rr * ur - rcr * uc) / (rr - rcr);
   }

   std::stringstream sstr;
   sstr << "left state " << std::endl;
   sstr << "left wave  ";
   if (l_w == 'r')
   {
      sstr << "    rarefaction" << std::endl;
   }
   else
   {
      sstr << "       shock   "
           << "  s = " << uls << std::endl;
   }
   sstr << "           "
        << "    r = " << rcl << std::endl;
   sstr << "contact    "
        << "    p = " << pc << " v = " << uc << std::endl;
   sstr << "           "
        << "    r = " << rcr << std::endl;
   sstr << "right wave ";
   if (r_w == 'r')
   {
      sstr << "    rarefaction" << std::endl;
   }
   else
   {
      sstr << "       shock   "
           << "  s = " << urs << std::endl;
   }
   sstr << "right state" << std::endl;

   return uc;
}

} // namespace riemann1D
