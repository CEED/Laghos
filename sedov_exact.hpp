#include <cmath>
#include <fstream>
#include <iostream>

// hardcoded input parameters;
// they can not be changed without changing the implementation.
const double sedov_rho0  = 1.;
const double sedov_gamma = 1.4;
// const double sedov_E0  = 1.;

// time at which the solution is evaluated
double sedov_t;
// position of the shock at time sedov_t
double sedov_rs;

void sedov_time(double t)
{
   sedov_t  = t;
   sedov_rs = 1.0040216061302775389 * sqrt(sedov_t);
}

// implementation of the secant method;
// solves  f(x) = fx
int invert_func_secant(double (*f)(double),
                       double a,
                       double b,
                       double fx,
                       double max_slope,
                       double &x,
                       int max_it)
{
   // g(y) := f(y) - fx
   // x_{n+1} = x_n - g(x_n) * (x_n - x_{n-1}) / (g(x_n) - g(x_{n-1}))
   //         = x_n - (f(x_n) - fx) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
   double xn = b;
   double dx = b - a;
   double fn = f(xn);
   double df = fn - f(a);

   int i;
   for (i = 0; i < max_it; i++)
   {
      if (df == 0.0)
      {
         break;
      }
      dx *= (fx - fn) / df;
      if (dx == 0.0)
      {
         break;
      }
      xn += dx;

      df = fn;
      fn = f(xn);
      df = fn - df;
   }

   x = xn;

   if (fabs(fn - fx) > 1.0e-14 * max_slope * fabs(fx))
   {
      return 1;
   }

   return 0;
}

double sedov_r2k(double eps)
{
   return (0.9319246811611120694 * eps * pow(1. - eps * eps / 36., -7. / 2.));
}

double sedov_epsr(double r)
{
   int no_conv;
   double eps;

   no_conv = invert_func_secant(sedov_r2k, 1.1, 1.0, pow(r / sqrt(sedov_t), 7.), 1.24, eps, 12);

   if (eps < 0.0)
   {
      return 0.0;
   }
   return eps;
}

double sedov_rhoeps(double eps)
{
   return (1.8273064355862722579 * pow(eps, 5. / 7.) * pow(0.4 + 1.44 / (2.4 - eps), 10. / 3.));
}

double sedov_rho(double r)
{
   if (r > sedov_rs)
   {
      return sedov_rho0;
   }

   return sedov_rhoeps(sedov_epsr(r));
}

double sedov_peps(double eps)
{
   return (0.065782785110333115709 * (6. + eps) * pow(0.2 + 0.72 / (2.4 - eps), 7. / 3.) / sedov_t);
}

double sedov_p(double r)
{
   if (r > sedov_rs)
   {
      return 0.0;
   }

   return sedov_peps(sedov_epsr(r));
}

double sedov_rh2k(double eps)
{
   return (15.472111057950348815 * eps / (6. + eps) * pow(0.2 + 0.72 / (2.4 - eps), 7. / 3.));
}

double sedov_epsrh(double rh)
{
   int no_conv;
   double eps;
   const double min_slope = 0.5116759451;
   double fx              = pow(rh, 2.) / sedov_t;
   double eps1;
   eps1 = (fx < min_slope) ? (fx / min_slope) : 1.0;

   no_conv = invert_func_secant(sedov_rh2k, 1.1 * eps1, eps1, fx, 2.08, eps, 12);

   return eps;
}

double sedov_rrh(double rh)
{
   if (rh > sedov_rs)
   {
      return rh;
   }

   return sqrt(sedov_t) * pow(sedov_r2k(sedov_epsrh(rh)), 1. / 7.);
}

double sedov_eeps(double eps) { return sedov_peps(eps) / ((sedov_gamma - 1.) * sedov_rhoeps(eps)); }

double sedov_e(double r)
{
   if (r > sedov_rs)
   {
      return 0.0;
   }

   return sedov_eeps(sedov_epsr(r));
}

double sedov_veps(double eps)
{
   return (0.35356380510848455185 * pow(eps, 1. / 7.) / sqrt(sedov_t * (6. - eps) / (6. + eps)));
}

double sedov_v(double r)
{
   if (r > sedov_rs)
   {
      return 0.0;
   }

   return sedov_veps(sedov_epsr(r));
}

double sedov_rho(const double *xyt)
{
   sedov_time(xyt[2]);
   return sedov_rho(sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]));
}

double sedov_e(const double *xyt)
{
   sedov_time(xyt[2]);
   return sedov_e(sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]));
}

double sedov_p(const double *xyt)
{
   sedov_time(xyt[2]);
   return sedov_p(sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]));
}

double sedov_phi_x(const double *xyt)
{
   sedov_time(xyt[2]);
   double rh = sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]);
   if (rh > 0.0)
   {
      return xyt[0] * sedov_rrh(rh) / rh;
   }
   return 0.0;
}

double sedov_phi_y(const double *xyt)
{
   sedov_time(xyt[2]);
   double rh = sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]);
   if (rh > 0.0)
   {
      return xyt[1] * sedov_rrh(rh) / rh;
   }
   return 0.0;
}

double sedov_vx(const double *xyt)
{
   sedov_time(xyt[2]);
   double r = sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]);
   if (r > 0.0)
   {
      return sedov_v(r) * xyt[0] / r;
   }
   return 0.0;
}

double sedov_vy(const double *xyt)
{
   sedov_time(xyt[2]);
   double r = sqrt(xyt[0] * xyt[0] + xyt[1] * xyt[1]);
   if (r > 0.0)
   {
      return sedov_v(r) * xyt[1] / r;
   }
   return 0.0;
}

//int main(void)
//{
//   std::ofstream datafile1("sedov_rho.dat");
//   std::ofstream datafile2("sedov_p.dat");
//   std::ofstream datafile3("sedov_rrh.dat");
//   std::ofstream datafile4("sedov_v.dat");

//   datafile1.precision(16);
//   datafile1.setf(std::ios::scientific);
//   datafile2.precision(16);
//   datafile2.setf(std::ios::scientific);
//   datafile3.precision(16);
//   datafile3.setf(std::ios::scientific);
//   datafile4.precision(16);
//   datafile4.setf(std::ios::scientific);
//   std::cout.precision(16);
//   std::cout.setf(std::ios::scientific);

//   double t;
//   std::cout << "Enter t = " << std::flush;
//   std::cin >> t;
//   sedov_time(t);
//   std::cout << "time:                   " << sedov_t << std::endl;
//   std::cout << "shock position:         " << sedov_rs << std::endl;
//   std::cout << "post-shock density:     " << sedov_rhoeps(1.) << std::endl;
//   std::cout << "post-shock pressure:    " << sedov_peps(1.) << std::endl;
//   std::cout << "pressure at the origin: " << sedov_peps(0.) << std::endl;
//   std::cout << "post-shock int. energy: " << sedov_eeps(1.) << std::endl;
//   std::cout << "post-shock velocity:    " << sedov_veps(1.) << std::endl;
//   const int n  = 8*1024;
//   const int ns = 3*n/4;
//   for (int i = 0; i <= n; i++)
//   {
//      double r = (double(i) / ns) * sedov_rs;

//      datafile1 << r << "  " << sedov_rho(r) << '\n';
//      datafile2 << r << "  " << sedov_p(r) << '\n';
//      datafile3 << r << "  " << sedov_rrh(r) << '\n';
//      datafile4 << r << "  " << sedov_v(r) << '\n';
//   }
//   datafile4.flush();
//   datafile4.close();
//   datafile3.flush();
//   datafile3.close();
//   datafile2.flush();
//   datafile2.close();
//   datafile1.flush();
//   datafile1.close();

//   return 0;
//}
