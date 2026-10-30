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

#include "laghos_solver.hpp"

// TODO: Do we want this to be included from mfem.hpp automatically now?
// Its non-inline definitions restrict it to this translation unit
#include "fem/dfem/doperator.hpp"
#include "linalg/kernels.hpp"

namespace mfem
{

namespace hydrodynamics
{

using future::FieldDescriptor;
using future::Gradient;
using future::Identity;
using future::Value;
using future::Weight;
using future::pretty_print;

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, bool vec)
{
   gf.HostRead();
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         const char* keys = (gf.FESpace()->GetMesh()->Dimension() == 2)
                            ? "mAcRjl" : "mmaaAcl";

         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys " << keys;
         if ( vec ) { sock << "vvv"; }
         sock << std::endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

void threshold(Vector &v)
{
   for (int i = 0; i < v.Size(); i++)
   {
      if (abs(v(i)) <= 1e-12)
      {
         v(i) = 0.0;
      }
   }
}

MFEM_HOST_DEVICE inline
real_t taylor_source(const Vector &x)
{
   return 3.0 / 8.0 * M_PI * ( cos(3.0*M_PI*x(0)) * cos(M_PI*x(1)) -
                               cos(M_PI*x(0))     * cos(3.0*M_PI*x(1)) );
};

MFEM_HOST_DEVICE inline
real_t smoothmin(real_t a, real_t b, real_t k = 1e6)
{
   return -1.0 / k * log(exp(-k * a) + exp(-k * b));
}

MFEM_HOST_DEVICE inline
real_t smoothmax(real_t a, real_t b, real_t k = 1e-6)
{
   return 0.5 * (a + b + sqrt((a-b)*(a-b) + k*k));
}

MFEM_HOST_DEVICE inline
real_t smoothabs(real_t x, real_t k = 1e-6)
{
   return sqrt(x * x + k);
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
tensor<T, n> shift(const tensor<T, n> &v, T s)
{
   tensor<T, n> sv;
   for (int i = 0; i < n; i++)
   {
      sv(i) = v(i) - s;
   }
   return sv;
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
T min(const tensor<T, n> &v)
{
   T min = std::numeric_limits<T>::min();
   for (int i = 0; i < n; i++)
   {
      if (v(i) < min)
      {
         min = v(i);
      }
   }
   return min;
}

template <typename T, int n>
MFEM_HOST_DEVICE inline
std::tuple<T, tensor<T, n>> sinvpm(const tensor<T, n, n> &A, int maxit, T tol)
{
   auto shift = [](const tensor<T, n, n> &A, const T& mu)
   {
      const auto I = mfem::future::IdentityMatrix<n>();
      tensor<T, n, n> B = A;
      B -= mu * I;
      return B;
   };

   auto As = shift(A, tol);
   auto mu = min(std::get<0>(eig(As)));
   tensor<T, n> x = {};
   x(0) = 1.0;
   x = x / norm(x);
   const auto Binv = inv(shift(A, mu));
   auto y = Binv * x;
   auto la = dot(y, x);

   for (int i = 0; i < maxit; i++)
   {
      // const auto err = norm(y - la * x) / norm(y);
      // if (err <= tol)
      // {
      //    break;
      // }
      x = y / norm(y);
      y = Binv * x;
      la = dot(y, x);
   }

   return {mu + 1.0 / la, x};
}

// Smooth transition between 0 and 1 for x in [-eps, eps].
MFEM_HOST_DEVICE inline
real_t smooth_step_01(real_t x, real_t eps)
{
   const real_t y = (x + eps) / (2.0 * eps);
   if (y < 0.0) { return 0.0; }
   if (y > 1.0) { return 1.0; }
   return (3.0 - 2.0 * y) * y * y;
}

MFEM_HOST_DEVICE inline
void ComputeMaterialProperties(const real_t &gamma, const real_t &rho,
                               const real_t &E, real_t &p, real_t &cs)
{
   p = (gamma - 1.0) * rho * E;
   cs = sqrt(gamma * (gamma - 1.0) * E);
}

// static inline MFEM_HOST_DEVICE
// std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2> >
// eig2(tensor<real_t, 2, 2> &A)
// {
//    return eig(A);
// }

// static inline MFEM_HOST_DEVICE
// std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2> >
// grad_eig2(
//    tensor<real_t, 2, 2> &A,
//    tensor<real_t, 2, 2> &dA)
// {
//    return grad_eig(A, dA);
// }

// void* __enzyme_register_derivative_eig[] =
// {
//    (std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>>*)eig2,
//    (std::tuple<tensor<real_t, 2>, tensor<real_t, 2, 2>>*)grad_eig2,
// };

// dfem-dev has no 3x3 calcsv: local copy from the dfem-laghos-dev-av branch,
// commit d911c400de "calcsv for 3x3", to be removed once it lands in dfem-dev
template <typename T> MFEM_HOST_DEVICE
T calcsv(const future::tensor<T, 3, 3> A, const int i)
{
   double mult;
   double a00, a01, a02, a10, a11, a12, a20, a21, a22;

   a00 = A(0, 0); a01 = A(0, 1); a02 = A(0, 2);
   a10 = A(1, 0); a11 = A(1, 1); a12 = A(1, 2);
   a20 = A(2, 0); a21 = A(2, 1); a22 = A(2, 2);

   {
      double d_max = fabs(a00);
      if (d_max < fabs(a01)) { d_max = fabs(a01); }
      if (d_max < fabs(a02)) { d_max = fabs(a02); }
      if (d_max < fabs(a10)) { d_max = fabs(a10); }
      if (d_max < fabs(a11)) { d_max = fabs(a11); }
      if (d_max < fabs(a12)) { d_max = fabs(a12); }
      if (d_max < fabs(a20)) { d_max = fabs(a20); }
      if (d_max < fabs(a21)) { d_max = fabs(a21); }
      if (d_max < fabs(a22)) { d_max = fabs(a22); }
      future::GetScalingFactor(d_max, mult);
   }

   a00 /= mult; a01 /= mult; a02 /= mult;
   a10 /= mult; a11 /= mult; a12 /= mult;
   a20 /= mult; a21 /= mult; a22 /= mult;

   // Build B = A^T A (symmetric positive semidefinite)
   const double b00 = a00*a00 + a10*a10 + a20*a20;
   const double b01 = a00*a01 + a10*a11 + a20*a21;
   const double b02 = a00*a02 + a10*a12 + a20*a22;
   const double b11 = a01*a01 + a11*a11 + a21*a21;
   const double b12 = a01*a02 + a11*a12 + a21*a22;
   const double b22 = a02*a02 + a12*a12 + a22*a22;

   double e0, e1, e2;

   // Closed-form eigenvalues for symmetric 3x3 matrix
   // See eigenvalue algorithm - stable trigonometric form
   const double p1 = b01*b01 + b02*b02 + b12*b12;
   if (p1 == 0.0)
   {
      // B is diagonal
      e0 = b00;
      e1 = b11;
      e2 = b22;
   }
   else
   {
      const double q = (b00 + b11 + b22) / 3.0;

      const double c00 = b00 - q;
      const double c11 = b11 - q;
      const double c22 = b22 - q;

      const double p2 = c00*c00 + c11*c11 + c22*c22 + 2.0*p1;
      const double p = std::sqrt(p2 / 6.0);

      // C = (1/p) * (B - q I)
      const double invp = 1.0 / p;
      const double c01 = b01 * invp;
      const double c02 = b02 * invp;
      const double c12 = b12 * invp;
      const double cc00 = c00 * invp;
      const double cc11 = c11 * invp;
      const double cc22 = c22 * invp;

      // r = det(C) / 2
      const double detC =
         cc00*(cc11*cc22 - c12*c12) -
         c01*(c01*cc22 - c12*c02) +
         c02*(c01*c12 - cc11*c02);
      double r = 0.5 * detC;

      // Clamp r into [-1, 1] to avoid NaNs from acos due to roundoff
      if (r < -1.0) { r = -1.0; }
      if (r >  1.0) { r =  1.0; }

      const double pi = 3.141592653589793238462643383279502884;
      const double phi = std::acos(r) / 3.0;

      // These are the eigenvalues of B, ordered as e0 >= e1 >= e2 in exact arithmetic
      e0 = q + 2.0*p*std::cos(phi);
      e2 = q + 2.0*p*std::cos(phi + 2.0*pi/3.0);
      e1 = 3.0*q - e0 - e2;
   }

   // Enforce non-negativity (B is PSD, but roundoff can make tiny negatives)
   if (e0 < 0.0) { e0 = 0.0; }
   if (e1 < 0.0) { e1 = 0.0; }
   if (e2 < 0.0) { e2 = 0.0; }

   // Select requested eigenvalue by sorting (robust even in nearly-degenerate cases)
   const double emax = fmax(e0, fmax(e1, e2));
   const double emin = fmin(e0, fmin(e1, e2));
   const double emid = (e0 + e1 + e2) - emax - emin;

   const double chosen = (i == 0) ? emax : ((i == 1) ? emid : emin);
   return std::sqrt(chosen) * mult;
}

template <int DIM>
MFEM_HOST_DEVICE inline
matd_t<DIM> qdata_setup(
   const matd_t<DIM> &dvdxi,
   const real_t &rho0,
   const matd_t<DIM> &J0,
   const matd_t<DIM> &J,
   const real_t &gamma,
   const real_t &E,
   const real_t &w,
   const real_t &h0,
   const real_t &order_v,
   const real_t &cfl,
   const bool &use_viscosity,
   const int &viscosity_type,
   const real_t &visc_q1,
   const real_t &visc_q2,
   real_t &dt_est)
{
   using matd = matd_t<DIM>;

   // constexpr real_t eps = 1e-12;
   // constexpr real_t vorticity_coeff = 1.0;
   real_t p, cs;
   const real_t detJ = det(J);
   const matd invJ = inv(J);
   matd stress{{{0.0}}};
   const real_t rho = rho0 * det(J0) / detJ;
   // The stock Laghos viscosity (type 1) also uses the stock clamp of E
   const bool stock_av = use_viscosity && viscosity_type == 1;
   const real_t Ez = stock_av ? fmax(0.0, E) : smoothmax(0.0, E, 1e-6);
   real_t dt_visc_coeff = 0.0;

   ComputeMaterialProperties(gamma, rho, Ez, p, cs);

   for (int d = 0; d < DIM; d++) { stress(d, d) = -p; }

   if (use_viscosity)
   {
      /*auto softstep = [](const real_t &width, const real_t x)
      {
         // Clamping to prevend under/overflow.
         // This does not affect smoothness, it's only for extreme values.
         // [-40,40] covers most practical values.
         //auto arg = std::max(std::min(x / (width + 1e-6), 40.0), -40.0);

         // Smooth transition from 0 to 1; symmetric around the zero.
         // width: controls the slope of the sigmoid; lower -> sharper.
         // value(0) = 0.5, value(+width) ~ 0.73, value(-width) ~ 0.27.
         //return 1.0 / (1.0 + std::exp(-arg));

         // Value at x=0.
         real_t y0 = width;
         real_t shift = 2.0 * width * atanh(1.0 - 2.0 * y0);
         return 0.5 + 0.5 * tanh(0.5 * (x-shift) / width);
      };*/

      // y0 is value at x=0.
      auto softstep7 = [](const real_t &width, const real_t y0, const real_t x)
      {
         real_t shift = 2.0 * width * atanh(1.0 - 2.0 * y0);
         return 0.5 + 0.5 * tanh(0.5 * (x-shift) / width);
      };

      auto softabs = [](const real_t &eps, const real_t x)
      {
         // Clamping to prevend under/overflow; doesn't affect smoothness.
         //auto e = std::max(eps, 1e-6_r);

         // Diffuse the kink at |x| ~ 0. Activates as |x| approaches eps.
         //return sqrt(x * x + e * e);

         return hypot(x, eps);
      };

      if (viscosity_type == 1)
      {
         // Stock Laghos tensor viscosity, see QUpdateBody in laghos_solver.cpp
         // (without the vorticity scaling, as for the stock Sedov problem)
         // The symmetric velocity gradient eigenvector of the smallest
         // eigenvalue gives the direction of maximal compression, which
         // defines the length scale h from the initial one h0
         const matd sgrad_v = sym(dvdxi * invJ);
         real_t sgrad_v_data[DIM*DIM], eig_val[DIM], eig_vec[DIM*DIM];
         for (int j = 0; j < DIM; j++)
         {
            for (int i = 0; i < DIM; i++) { sgrad_v_data[i + DIM*j] = sgrad_v(i, j); }
         }
         if constexpr (DIM == 1)
         {
            eig_val[0] = sgrad_v_data[0];
            eig_vec[0] = 1.0;
         }
         else { kernels::CalcEigenvalues<DIM>(sgrad_v_data, eig_val, eig_vec); }
         vecd_t<DIM> compr_dir;
         for (int k = 0; k < DIM; k++) { compr_dir(k) = eig_vec[k]; }
         const vecd_t<DIM> ph_dir = (J * inv(J0)) * compr_dir;
         const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // Measure of maximal compression
         const real_t mu = eig_val[0];
         // Quadratic term, then a smooth "if (mu < 0) += 0.5 rho h cs"
         constexpr real_t eps = 1e-12;
         real_t visc_coeff = 2.0 * rho * h * h * fabs(mu);
         visc_coeff += 0.5 * rho * h * cs *
                       (1.0 - smooth_step_01(mu - 2.0 * eps, eps));
         stress += visc_coeff * sgrad_v;
         dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 2)
      {
         // Default Laghos viscosity.
         // auto symdvdx = sym(dvdxi * invJ);
         // auto [eigvals, eigvecs] = eig(symdvdx);

         // // Measure of maximal compression.
         // const real_t mu = eigvals(0);
         // vecd compr_dir = get_col(eigvecs, 0);

         // auto ph_dir = (J * inv(J0)) * compr_dir;
         // const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // auto visc_coeff = 2.0 * rho * h * h * fabs(mu);
         // visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
         //               (1.0 - softstep(eps, mu - 2.0 * eps));
         // stress += visc_coeff * symdvdx;
         // dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 21)
      {
         MFEM_ABORT("works only in 2D!");

         // Default Laghos viscosity through a power method to
         // get the measure of maximal compression.
         // auto symdvdx = sym(dvdxi * invJ);
         // auto [mu, compr_dir] = sinvpm(symdvdx, 10, 1e-12);

         // auto ph_dir = (J * inv(J0)) * compr_dir;
         // const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // auto visc_coeff = 2.0 * rho * h * h * softabs(1e-6, mu);
         // visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
         //               (1.0 - softstep(eps, mu - 2.0 * eps));
         // stress += visc_coeff * symdvdx;
         // dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 22)
      {
         MFEM_ABORT("works only in 2D!");

         // const auto delta = 0.2 * cs;

         // auto symdvdx = sym(dvdxi * invJ);
         // auto [eigvals, eigvecs] = eig2(symdvdx);

         // // Measure of maximal compression.
         // const real_t mu = eigvals(0);
         // vecd compr_dir = get_col(eigvecs, 0);
         // for (int i = 0; i < compr_dir.first_dim; i++)
         // {
         //    compr_dir(i) = softstep(delta, compr_dir(i));
         // }

         // auto ph_dir = (J * inv(J0)) * compr_dir;
         // const real_t h = h0 * norm(ph_dir) / norm(compr_dir);
         // auto visc_coeff = 2.0 * rho * h * h * softabs(delta, mu);
         // visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
         //               (1.0 - softstep(eps, mu - 2.0 * eps));
         // if (!std::isfinite(visc_coeff)) { out << "err\n"; exit(1); }
         // stress += visc_coeff * symdvdx;
         // dt_visc_coeff = visc_coeff;
      }
      else if (viscosity_type == 4)
      {
         MFEM_ABORT("works only in 2D!");

         // Viscosity type 4 from the paper (use all eigenvalues).
         // auto symdvdx = sym(dvdxi * invJ);
         // auto [lam, s] = eig(symdvdx);
         // const auto delta = 0.2 * cs;

         // for (int k = 0; k < DIMENSION; k++)
         // {
         //    const auto ph_dir = (J * inv(J0)) * s(k);
         //    const real_t h = h0 * norm(ph_dir) / norm(s(k));
         //    auto visc_coeff = 2.0 * rho * h * h * softabs(delta, lam(k));
         //    visc_coeff += 0.5 * rho * h * cs * vorticity_coeff *
         //                  (1.0 - softstep(delta, lam(k) - 2.0 * eps));
         //    stress += visc_coeff * symdvdx;
         //    dt_visc_coeff += visc_coeff;
         // }
      }
      else if (viscosity_type == 7)
      {
         // Smoother viscosity for well-defined gradients.

         const auto dvdx = dvdxi * inv(J);
         const auto h = h0 * pow(det(J) / det(J0), 1.0 / static_cast<real_t>(DIM));
         const auto delta_v = h * tr(dvdx);

         // Coarses mesh h for the domain.
         const real_t h_coarse = 1.0;
         real_t dv_scale = 2.0 * h / h_coarse * cs;

         // Smooth activation switch.
         // Motivation: in compression, delta_v < 0.
         //             in shocks, delta_v starts to be comparable to cs.
         // -> this viscosity becomes noticable when delta_v > dv_scale.
         const auto psi = softstep7(dv_scale, 0.5, -delta_v);

         // Smooths the kink at |delta_v| ~ 0 with correct units scaling.
         // The smoothing becomes active when |delta_v| approaches dv_scale.
         const auto abs_delta_v = softabs(dv_scale, delta_v);

         const auto mu = rho * h * (visc_q2 * abs_delta_v + psi * visc_q1 * cs);
         stress += mu * sym(dvdx);

         dt_visc_coeff = mu;
      }
      else { MFEM_ABORT("wrong viscosity option"); }
   }

   if (rho < 0.0)
   {
      //MFEM_ABORT("negative density on quadrature point \n" "detJ = " << detJ);
      //exit(1);
      dt_est = 0.0;
   }

   if (detJ < 0.0)
   {
      //MFEM_ABORT("inverted element detected in qdata_setup");
      // This will force repetition of the step with smaller dt.
      dt_est = 0.0;
   }
   else
   {
      real_t sv = calcsv(J, DIM-1);

      const real_t hmin = sv / static_cast<real_t>(order_v);
      const real_t ihmin = 1.0 / hmin;
      const real_t irhoihminsq = ihmin * ihmin / rho;
      const real_t idt = cs / hmin + 2.5 * dt_visc_coeff * irhoihminsq;

      if (idt > 0.0)
      {
         dt_est = fmin(dt_est, cfl / idt);
      }
   }

   matd stressJiT = stress * transpose(invJ) * detJ * w;
   return stressJiT;
}

template <int DIM>
struct TimeStepEstimateQFunction
{
   using matd = matd_t<DIM>;

   TimeStepEstimateQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                   const matd &J, const real_t &gamma, const real_t &E,
                   const real_t &w, real_t &dt_est) const
   {
      dt_est = std::numeric_limits<real_t>::infinity();

      // Reuse qdata_setup() for the dt estimate logic; ignore its stress output.
      (void)qdata_setup<DIM>(
         dvdxi, rho0, J0, J, gamma, E, w,
         external_data[EXT_DATA_IDX::H0],
         external_data[EXT_DATA_IDX::ORDER_VEL],
         external_data[EXT_DATA_IDX::CFL],
         static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
         static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
         external_data[EXT_DATA_IDX::VISC_Q1],
         external_data[EXT_DATA_IDX::VISC_Q2],
         dt_est);
   }

   real_t *external_data;
};

template <int DIM>
struct UpdateQuadratureDataQFunction
{
   using matd = matd_t<DIM>;

   UpdateQuadratureDataQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                   const matd &J, const real_t &gamma, const real_t &E,
                   const real_t &w, matd &stressJiT) const
   {
      // Do not update the global dt estimate here; that will be computed
      // through a proper reduction over quadrature points.
      real_t dt_est_dummy = std::numeric_limits<real_t>::infinity();
      stressJiT =
         qdata_setup<DIM>(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
            external_data[EXT_DATA_IDX::VISC_Q1],
            external_data[EXT_DATA_IDX::VISC_Q2],
            dt_est_dummy);
   }

   real_t *external_data;
};

template <int DIM>
class MomentumQFunction
{
   using matd = matd_t<DIM>;

public:
   MomentumQFunction(real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   void operator()(const matd &dvdxi, const real_t &rho0, const matd &J0,
                   const matd &J, const real_t &gamma, const real_t &E,
                   const real_t &w, matd &stressJiT) const
   {
      stressJiT =
         qdata_setup<DIM>(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
            external_data[EXT_DATA_IDX::VISC_Q1],
            external_data[EXT_DATA_IDX::VISC_Q2],
            dt_est_dummy);
   }

   mutable real_t dt_est_dummy = std::numeric_limits<real_t>::infinity();
   real_t *external_data;
};

template <int DIM>
class EnergyConservationQFunction
{
   using matd = matd_t<DIM>;

public:
   EnergyConservationQFunction(const real_t *external_data) :
      external_data(external_data) {}

   MFEM_HOST_DEVICE inline
   auto operator()(
      const matd &dvdxi,
      const real_t &rho0,
      const matd &J0,
      const matd &J,
      const real_t &gamma,
      const real_t &E,
      const real_t &w,
      real_t &de) const
   {
      auto stressJiT =
         qdata_setup<DIM>(
            dvdxi, rho0, J0, J, gamma, E, w,
            external_data[EXT_DATA_IDX::H0],
            external_data[EXT_DATA_IDX::ORDER_VEL],
            external_data[EXT_DATA_IDX::CFL],
            static_cast<bool>(external_data[EXT_DATA_IDX::VISCOSITY_FLAG]),
            static_cast<int>(external_data[EXT_DATA_IDX::VISCOSITY_TYPE]),
            external_data[EXT_DATA_IDX::VISC_Q1],
            external_data[EXT_DATA_IDX::VISC_Q2],
            dt_est);
      de = ddot(stressJiT, dvdxi);
   }

   mutable real_t dt_est = std::numeric_limits<real_t>::infinity();
   const real_t *external_data;
};

void LagrangianHydroOperator::LineSearchNewtonSolver::ProcessNewState(
   const Vector &x) const
{
   if (problem == 0) { return; }

   s_new.SetSize(x.Size());
   s_new = x;
   s_new *= hydro.residual->dt;
   s_new += hydro.residual->x;

   real_t min_detJ = hydro.ComputeMinDet(s_new);
   if (min_detJ <= 0.0)
   {
      out << "ProcessNewState negative detJ = " << min_detJ << "\n";
   }
}

real_t LagrangianHydroOperator::LineSearchNewtonSolver::ComputeScalingFactor(
   const Vector &x, const Vector &b) const
{
   if (problem == 0) { return 1.0; }

   const bool have_b = (b.Size() == Height());
   real_t lambda = 1.0;  // Initial step length
   x_new.SetSize(x.Size());
   r_new.SetSize(x.Size());
   s_new.SetSize(x.Size());

   const real_t initial_norm = Norm(r);  // Current residual norm

   grad->Mult(c, r_new);
   const real_t grad_norm = Norm(r_new);     // Gradient norm

   for (int i = 0; i < max_line_iter; i++)
   {
      // Try step: x_new = x - lambda * c
      add(x, -lambda, c, x_new);

      // Before calling Mult on the operator we have to check mesh
      // validity by computing the minimum determinant of the mesh
      // Jacobian.
      s_new = x_new;
      s_new *= hydro.residual->dt;
      s_new += hydro.residual->x;
      real_t min_detJ = hydro.ComputeMinDet(s_new);
      if (min_detJ <= 0.0)
      {
         out << "linesearch found negative detJ = " << min_detJ << "\n";
         lambda *= beta;
         continue;
      }

      // Evaluate residual at new point
      oper->Mult(x_new, r_new);
      if (have_b) { subtract(r_new, b, r_new); }

      const real_t new_norm = Norm(r_new);

      if (new_norm <= initial_norm - alpha * lambda * grad_norm)
      {
         // Found acceptable step
         // if (Mpi::Root())
         // { out << "linesearch good lambda: " << lambda << "\n"; }
         return lambda;
      }

      // Backtrack
      lambda *= beta;
   }

   if (Mpi::Root()) { out << ">>> linesearch didn't converge\n"; }
   return 0.0;
}

// DerivativeOperator::Assemble returns a new matrix, owned by the caller
static std::unique_ptr<HypreParMatrix> AssembleMatrix(DerivativeOperator &d)
{
   HypreParMatrix *A = nullptr;
   d.Assemble(A);
   return std::unique_ptr<HypreParMatrix>(A);
}

void LagrangianHydroOperator::LagrangianHydroJacobianOperator::Mult(
   const Vector &u, Vector &y) const
{
   w = u;
   Vector wx, wv, we;
   wx.MakeRef(w, 0, H1tsize);
   wv.MakeRef(w, H1tsize, H1tsize);
   we.MakeRef(w, 2*H1tsize, L2tsize);

   Vector zx, zv, ze;
   zx.MakeRef(z, 0, H1tsize);
   zv.MakeRef(z, H1tsize, H1tsize);
   ze.MakeRef(z, 2*H1tsize, L2tsize);
   MultiVector Zv{zv}, Ze{ze};

   Vector yx, yv, ye;
   yx.MakeRef(y, 0, H1tsize);
   yv.MakeRef(y, H1tsize, H1tsize);
   ye.MakeRef(y, 2*H1tsize, L2tsize);

   // position
   yx = wv;
   yx *= -h;
   yx += wx;

   // velocity
   // wv.SetSubVector(hydro.ess_tdof, 0.0);
   dRvdx->Mult(wx, Zv);
   if (zv.CheckFinite() != 0)
   {
      pretty_print(zv);
   }
   MFEM_VERIFY(zv.CheckFinite() == 0, "err");
   zv *= h;
   yv = zv;
   dRvdv->Mult(wv, Zv);
   MFEM_VERIFY(zv.CheckFinite() == 0, "err");
   zv *= h;
   yv += zv;
   // hydro.Mv.TrueAddMult(wv, yv);
   Vector wvc, yvc;
   for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
   {
      wvc.MakeRef(wv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      yvc.MakeRef(yv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      hydro.Mv->FullAddMult(wvc, yvc);
      yvc.SyncAliasMemory(yv);
   }
   yv.SyncAliasMemory(y);

   dRvde->Mult(we, Zv);
   MFEM_VERIFY(zv.CheckFinite() == 0, "err");
   zv *= h;
   yv += zv;
   yv.SetSubVector(hydro.ess_tdof, 0.0);

   // for (int i = 0; i < hydro.ess_tdof.Size(); i++)
   // {
   //    // yv(hydro.ess_tdof[i]) = uv(hydro.ess_tdof[i]);
   //    yv(hydro.ess_tdof[i]) = 0.0;
   // }
   // yv = 0.0;

   // energy
   //                          [ wx ]
   // [ dRe/dx dRe/dv dRe/de ] [ wv ]
   //                          [ we ]
   //

   dRedx->Mult(wx, Ze);
   MFEM_VERIFY(ze.CheckFinite() == 0, "err");
   if (problem == 0)
   {
      Vector ts(L2tsize);
      MultiVector Ts{ts};
      dTaylorSourcedx->Mult(wx, Ts);
      ze += ts;
      MFEM_VERIFY(ze.CheckFinite() == 0, "err");
   }
   ze *= -h;
   ye = ze;

   dRedv->Mult(wv, Ze);
   MFEM_VERIFY(ze.CheckFinite() == 0, "err");
   ze *= -h;
   ye += ze;

   dRede->Mult(we, Ze);
   MFEM_VERIFY(ze.CheckFinite() == 0, "err");
   ze *= -h;
   // OLD hydro.Me.TrueAddMult(we, ze);
   hydro.Me->FullAddMult(we, ze);
   MFEM_VERIFY(ze.CheckFinite() == 0, "err");

   ye += ze;

   yx.SyncAliasMemory(y);
   yv.SyncAliasMemory(y);
   ye.SyncAliasMemory(y);
}

MemoryClass
LagrangianHydroOperator::LagrangianHydroJacobianOperator::GetMemoryClass() const
{
   return Device::GetDeviceMemoryClass();
}

void LagrangianHydroOperator::Preconditioner::SetRebuildFlag(bool flag)
{
   rebuild = flag;
}

void LagrangianHydroOperator::Preconditioner::SetOperator(const Operator &op)
{
   if (!rebuild)
   {
      return;
   }

   jacobian = dynamic_cast<const LagrangianHydroJacobianOperator*>(&op);
   MFEM_VERIFY(jacobian != nullptr, "Preconditioner error.");

   auto comm = hydro.H1.GetComm();
   const real_t h = jacobian->h;

   HYPRE_BigInt *tdof_offsets = hydro.H1.GetTrueDofOffsets();

   // First row
   // Rx = x - h * v
   // yx = I * wx - h I * wv
   SparseMatrix dRxdx_diag(hydro.H1.GetTrueVSize());
   for (int i = 0; i < dRxdx_diag.Height(); i++)
   {
      dRxdx_diag.Set(i, i, 1.0);
   }
   dRxdx_diag.Finalize();
   HypreParMatrix dRxdx_mat(comm, hydro.H1.GlobalTrueVSize(),
                            tdof_offsets, &dRxdx_diag);

   // dRvdv = (Mv + h dF/dv)
   auto dRvdv_ptr = AssembleMatrix(*jacobian->dRvdv);
   HypreParMatrix &dRvdv_mat = *dRvdv_ptr;
   HypreParMatrix *Mv_hdRvdv_mat = Add(1.0, hydro.Mv_mat, h, dRvdv_mat);
   auto tmp2 = Mv_hdRvdv_mat->EliminateRowsCols(hydro.ess_tdof);
   delete tmp2;

   // dRede = Me - h * dF^T/de
   auto dRede_ptr = AssembleMatrix(*jacobian->dRede);
   HypreParMatrix &dRede_mat = *dRede_ptr;
   HypreParMatrix *Me_hdRede_mat = Add(1.0, hydro.Me_mat, -h, dRede_mat);

   if (hydro.preconditioner_type == PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG)
   {
      // if (Mpi::Root()) { out << "building pc\n"; }
      vv_mat.reset(Mv_hdRvdv_mat);
      amg_v.reset(new HypreBoomerAMG(*vv_mat));
      amg_v->SetPrintLevel(0);

      ee_mat.reset(Me_hdRede_mat);
      amg_e.reset(new HypreBoomerAMG(*ee_mat));
      amg_e->SetPrintLevel(0);
   }

   else if (hydro.preconditioner_type == PRECONDITIONER_TYPE::SUPERLU)
   {
#ifdef MFEM_USE_SUPERLU
      // dRxdv = -h * I
      SparseMatrix dRxdv_diag(hydro.H1.GetTrueVSize());
      for (int i = 0; i < dRxdv_diag.Height(); i++)
      {
         dRxdv_diag.Set(i, i, -h);
      }

      for (int i = 0; i < hydro.ess_tdof.Size(); i++)
      {
         dRxdv_diag.Set(hydro.ess_tdof[i], hydro.ess_tdof[i], 0.0);
      }
      dRxdv_diag.Finalize();

      HypreParMatrix dRxdv_mat(comm, hydro.H1.GlobalTrueVSize(),
                               tdof_offsets, &dRxdv_diag);

      // Second row
      // Rv = Mv * v + F * I
      // yv = (Mv + h * dF/dv) * wv + h * dF/dx * wx + h * dF/de * we

      // dRvdx = h * dF/dx
      auto dRvdx_ptr = AssembleMatrix(*jacobian->dRvdx);
      HypreParMatrix &dRvdx_mat = *dRvdx_ptr;
      dRvdx_mat.EliminateRows(hydro.ess_tdof);
      dRvdx_mat *= h;

      // dRvde = h * dF/de
      auto dRvde_ptr = AssembleMatrix(*jacobian->dRvde);
      HypreParMatrix &dRvde_mat = *dRvde_ptr;
      dRvde_mat.EliminateRows(hydro.ess_tdof);
      dRvde_mat *= h;

      // Third row
      // Re = Me * e - F^T
      // ye = (Me - h * dF/de) * we - h * dF/dx * wx - h * dF/dv * wv

      // dRedx = -h * dF^T/dx
      auto dRedx_ptr = AssembleMatrix(*jacobian->dRedx);
      HypreParMatrix &dRedx_mat = *dRedx_ptr;

      if (problem == 0)
      {
         auto dTaylorSourcedx_ptr = AssembleMatrix(*jacobian->dTaylorSourcedx);
         HypreParMatrix &dTaylorSourcedx_mat = *dTaylorSourcedx_ptr;
         dRedx_mat.Add(1.0, dTaylorSourcedx_mat);
      }

      dRedx_mat *= -h;

      // dRedv = -h * dF^T/dv
      auto dRedv_ptr = AssembleMatrix(*jacobian->dRedv);
      HypreParMatrix &dRedv_mat = *dRedv_ptr;
      auto tmp1 = dRedv_mat.EliminateCols(hydro.ess_tdof);
      delete tmp1;
      dRedv_mat *= -h;

      Array2D<const HypreParMatrix*> blocks(3, 3);
      blocks = nullptr;
      blocks(0, 0) = &dRxdx_mat;
      blocks(0, 1) = &dRxdv_mat;
      blocks(1, 0) = &dRvdx_mat;
      blocks(1, 1) = Mv_hdRvdv_mat;
      blocks(1, 2) = &dRvde_mat;
      blocks(2, 0) = &dRedx_mat;
      blocks(2, 1) = &dRedv_mat;
      blocks(2, 2) = Me_hdRede_mat;

      superlu_solver.reset(new SuperLUSolver(MPI_COMM_WORLD, 1));
      block_hypre.reset(HypreParMatrixFromBlocks(blocks, nullptr));
      superlu_mat.reset(new SuperLURowLocMatrix(*block_hypre));
      superlu_solver->SetSymmetricPattern(false);
      superlu_solver->SetOperator(*superlu_mat);
      superlu_solver->SetPrintStatistics(false);
#else
      MFEM_ABORT("MFEM is not built with SuperLU");
#endif
   }

   rebuild = false;
}

void LagrangianHydroOperator::Preconditioner::Mult(const Vector &x,
                                                   Vector &y) const
{
   if (hydro.preconditioner_type == PRECONDITIONER_TYPE::BLOCK_DIAGONAL_AMG)
   {
      w = x;
      Vector wx, wv, we;
      wx.MakeRef(w, 0, hydro.H1.GetTrueVSize());
      wv.MakeRef(w, hydro.H1.GetTrueVSize(), hydro.H1.GetTrueVSize());
      we.MakeRef(w, 2*hydro.H1.GetTrueVSize(), hydro.L2.GetTrueVSize());

      Vector yx, yv, ye;
      yx.MakeRef(y, 0, hydro.H1.GetTrueVSize());
      yv.MakeRef(y, hydro.H1.GetTrueVSize(), hydro.H1.GetTrueVSize());
      ye.MakeRef(y, 2*hydro.H1.GetTrueVSize(), hydro.L2.GetTrueVSize());

      yx = wx;
      amg_v->Mult(wv, yv);
      amg_e->Mult(we, ye);
   }
   else if (hydro.preconditioner_type == PRECONDITIONER_TYPE::SUPERLU)
   {
#ifdef MFEM_USE_SUPERLU
      superlu_solver->Mult(x, y);
#else
      MFEM_ABORT("MFEM is not built with SuperLU");
#endif
   }
}

void LagrangianHydroOperator::LagrangianHydroResidualOperator::Mult(
   const Vector &k, Vector &R) const
{
   u = k;
   u *= dt;
   u += x;

   auto kptr = const_cast<Vector*>(&k);
   Vector kx, kv, ke;
   kx.MakeRef(*kptr, 0, H1tsize);
   kv.MakeRef(*kptr, H1tsize, H1tsize);
   ke.MakeRef(*kptr, 2*H1tsize, L2tsize);

   Vector ux, uv, ue;
   ux.MakeRef(u, 0, H1tsize);
   uv.MakeRef(u, H1tsize, H1tsize);
   ue.MakeRef(u, 2*H1tsize, L2tsize);

   Vector Rx, Rv, Re;
   Rx.MakeRef(R, 0, H1tsize);
   Rv.MakeRef(R, H1tsize, H1tsize);
   Re.MakeRef(R, 2*H1tsize, L2tsize);

   Vector ux_l, uv_l, ue_l;
   ux_l.MakeRef(u_l, 0, H1vsize);
   uv_l.MakeRef(u_l, H1vsize, H1vsize);
   ue_l.MakeRef(u_l, 2*H1vsize, L2vsize);

   hydro.H1.GetProlongationMatrix()->Mult(ux, ux_l);
   hydro.H1.GetProlongationMatrix()->Mult(uv, uv_l);
   hydro.L2.GetProlongationMatrix()->Mult(ue, ue_l);

   hydro.UpdateMesh(ux_l);
   hydro.mesh_nodes.SyncMemory(ux_l);

   Rx = kx;
   Rx -= uv;

   // out << "in qupdate\n";
   // pretty_print(ux_l);
   hydro.qdata_is_current = false;
   hydro.UpdateQuadratureData(u_l);

   MultiVector RV{Rv};
   hydro.momentum_pa->Mult(MultiVector{hydro.qdata->stressp}, RV);

   // hydro.Mv.TrueAddMult(kv, Rv);
   Vector kvc, Rvc;
   for (int c = 0; c < hydro.H1.GetMesh()->Dimension(); c++)
   {
      kvc.MakeRef(kv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      Rvc.MakeRef(Rv, c*hydro.H1c.GetTrueVSize(), hydro.H1c.GetTrueVSize());
      hydro.Mv->FullAddMult(kvc, Rvc);
      Rvc.SyncAliasMemory(Rv);
   }
   Rv.SyncAliasMemory(R);

   Rv.SetSubVector(hydro.ess_tdof, 0.0);

   MultiVector RE{Re};
   hydro.energy_conservation_pa->Mult(MultiVector{uv_l, hydro.qdata->stressp},
                                      RE);

   Re.Neg();

   if (problem == 0)
   {
      MultiVector ES{e_source_t};
      hydro.taylor_source_mf->Mult(MultiVector{ux}, ES);
      Re -= e_source_t;
   }

   // hydro.Me.TrueAddMult(ke, Re);
   hydro.Me->FullAddMult(ke, Re);

   Rx.SyncAliasMemory(R);
   Rv.SyncAliasMemory(R);
   Re.SyncAliasMemory(R);
}

Operator& LagrangianHydroOperator::LagrangianHydroResidualOperator::GetGradient(
   const Vector &k) const
{
   u = k;
   u *= dt;
   u += x;

   auto kptr = const_cast<Vector*>(&k);
   Vector kx, kv, ke;
   kx.MakeRef(*kptr, 0, H1tsize);
   kv.MakeRef(*kptr, H1tsize, H1tsize);
   ke.MakeRef(*kptr, 2*H1tsize, L2tsize);

   Vector ux, uv, ue;
   ux.MakeRef(u, 0, H1tsize);
   uv.MakeRef(u, H1tsize, H1tsize);
   ue.MakeRef(u, 2*H1tsize, L2tsize);

   if (fd_gradient)
   {
      fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
      return *fd_jacobian;
   }
   else
   {
      // The derivatives take their state as T-vectors, in the inputs order
      Vector rho0_t(L2tsize), x0_t(H1tsize),
             material_t(hydro.material.ParFESpace()->GetTrueVSize());
      hydro.L2.GetRestrictionMatrix()->Mult(hydro.rho0, rho0_t);
      hydro.H1.GetRestrictionMatrix()->Mult(hydro.x0, x0_t);
      hydro.material.ParFESpace()->GetRestrictionMatrix()->Mult(hydro.material,
                                                                material_t);
      const MultiVector Rv_state{uv, rho0_t, x0_t, ux, material_t, ue};
      const MultiVector Re_state{ue, uv, rho0_t, x0_t, ux, material_t};

      auto dRvdx = hydro.momentum_mf->GetDerivative(COORDINATES, Rv_state);

      auto dRvdv = hydro.momentum_mf->GetDerivative(VELOCITY, Rv_state);

      auto dRvde = hydro.momentum_mf->GetDerivative(SPECIFIC_INTERNAL_ENERGY,
                                                    Rv_state);

      auto dRedx = hydro.energy_conservation_mf->GetDerivative(COORDINATES,
                                                               Re_state);

      auto dRedv = hydro.energy_conservation_mf->GetDerivative(VELOCITY,
                                                               Re_state);

      auto dRede = hydro.energy_conservation_mf->GetDerivative(
                      SPECIFIC_INTERNAL_ENERGY, Re_state);

      auto dTaylorSourcedx = hydro.taylor_source_mf->GetDerivative(
                                COORDINATES, MultiVector{ux});

      jacobian = std::make_shared<LagrangianHydroJacobianOperator>(
                    hydro, dRvdx, dRvdv, dRvde, dRedx, dRedv, dRede, dTaylorSourcedx, dt);

      if (dump_jacobians > 0)
      {
         auto ess_tdof_backup(hydro.ess_tdof);
         hydro.ess_tdof.SetSize(0);

         out << "\ndumping jacobians\n";
         std::ofstream jvpmat("jvpmat.m");
         jacobian->PrintMatlab(jvpmat);
         jvpmat.close();

         fd_jacobian = std::make_shared<FDJacobian>(*this, k, 1e-8);
         std::ofstream fdjacmat("fdjacmat.m");
         fd_jacobian->PrintMatlab(fdjacmat);
         fdjacmat.close();

         // hydro.preconditioner->SetOperator(*jacobian);
         // std::ofstream jprecmat("jprecmat.m");
         // hydro.preconditioner->block_hypre->PrintMatlab(jprecmat);
         // jprecmat.close();

         if (dump_jacobians == 1)
         {
            exit(0);
         }

         hydro.ess_tdof = ess_tdof_backup;
      }

      if (hydro.use_petsc)
      {
#ifdef MFEM_USE_PETSC
         const real_t h = jacobian->h;
         auto comm = hydro.H1.GetComm();

         // Build the assembled PETSc Jacobian.
         //
         // Layout:
         // [ x ]  size H1tsize
         // [ v ]  size H1tsize
         // [ e ]  size L2tsize

         const int H1tsize = hydro.H1.GetTrueVSize();
         const int L2tsize = hydro.L2.GetTrueVSize();

         //
         // Build diagonal blocks through HypreParMatrix to guarantee a
         // consistent parallel layout.
         //

         HYPRE_BigInt *H1_tdof_offsets = hydro.H1.GetTrueDofOffsets();

         // Rxdx.
         SparseMatrix dRxdx_diag(hydro.H1.GetTrueVSize());
         for (int i = 0; i < dRxdx_diag.Height(); i++)
         {
            dRxdx_diag.Set(i, i, 1.0);
         }
         dRxdx_diag.Finalize();
         HypreParMatrix dRxdx_mat(comm, hydro.H1.GlobalTrueVSize(),
                                  H1_tdof_offsets, &dRxdx_diag);
         PetscParMatrix dRxdx_petsc(&dRxdx_mat);

         // Rxdv.
         SparseMatrix dRxdv_diag(hydro.H1.GetTrueVSize());
         for (int i = 0; i < dRxdv_diag.Height(); i++)
         {
            dRxdv_diag.Set(i, i, -h);
         }
         for (int i = 0; i < hydro.ess_tdof.Size(); i++)
         {
            dRxdv_diag.Set(hydro.ess_tdof[i], hydro.ess_tdof[i], 0.0);
         }
         dRxdv_diag.Finalize();
         HypreParMatrix dRxdv_mat(comm, hydro.H1.GlobalTrueVSize(),
                                  H1_tdof_offsets, &dRxdv_diag);
         PetscParMatrix dRxdv_petsc(&dRxdv_mat);

         // Rvdx.
         auto dRvdx_ptr = AssembleMatrix(*jacobian->dRvdx);
         HypreParMatrix &dRvdx_mat = *dRvdx_ptr;
         dRvdx_mat.EliminateRows(hydro.ess_tdof);
         PetscParMatrix dRvdx_petsc(&dRvdx_mat);
         dRvdx_petsc *= h;

         // Rvdv.
         auto dRvdv_ptr = AssembleMatrix(*jacobian->dRvdv);
         HypreParMatrix &dRvdv_mat = *dRvdv_ptr;
         ParBilinearForm Mv(&hydro.H1);
         Mv.AddDomainIntegrator(new VectorMassIntegrator(hydro.rho0_coeff,
                                                         &hydro.ir));
         Mv.Assemble();
         Mv.Finalize();
         HypreParMatrix Mv_mat;
         Mv.FormSystemMatrix(mfem::Array<int>(), Mv_mat);
         HypreParMatrix *dRvdv_combined = Add(1.0, Mv_mat, h, dRvdv_mat);
         auto tmp0 = dRvdv_combined->EliminateRowsCols(hydro.ess_tdof);
         delete tmp0;
         PetscParMatrix dRvdv_petsc(dRvdv_combined);
         delete dRvdv_combined;

         // Rvde.
         auto dRvde_ptr = AssembleMatrix(*jacobian->dRvde);
         HypreParMatrix &dRvde_mat = *dRvde_ptr;
         dRvde_mat.EliminateRows(hydro.ess_tdof);
         PetscParMatrix dRvde_petsc(&dRvde_mat);
         dRvde_petsc *= h;

         // Redx.
         auto dRedx_ptr = AssembleMatrix(*jacobian->dRedx);
         HypreParMatrix &dRedx_mat = *dRedx_ptr;
         PetscParMatrix dRedx_petsc(&dRedx_mat);
         if (problem == 0)
         {
            auto dTaylorSourcedx_ptr = AssembleMatrix(*jacobian->dTaylorSourcedx);
            HypreParMatrix &dTaylorSourcedx_mat = *dTaylorSourcedx_ptr;
            PetscParMatrix dTaylorSourcedx_petsc(&dTaylorSourcedx_mat);
            dRedx_petsc += dTaylorSourcedx_petsc;
         }
         dRedx_petsc *= -h;

         // Redv.
         auto dRedv_ptr = AssembleMatrix(*jacobian->dRedv);
         HypreParMatrix &dRedv_mat = *dRedv_ptr;
         auto tmp1 = dRedv_mat.EliminateCols(hydro.ess_tdof);
         delete tmp1;
         PetscParMatrix dRedv_petsc(&dRedv_mat);
         dRedv_petsc *= -h;

         // Rede.
         auto dRede_ptr = AssembleMatrix(*jacobian->dRede);
         HypreParMatrix &dRede_mat = *dRede_ptr;
         PetscParMatrix dRede_petsc(&dRede_mat);
         ParBilinearForm Me(&hydro.L2);
         Me.AddDomainIntegrator(new MassIntegrator(hydro.rho0_coeff,
                                                   &hydro.ir));
         Me.Assemble();
         Me.Finalize();
         HypreParMatrix Me_mat;
         Me.FormSystemMatrix(mfem::Array<int>(), Me_mat);
         PetscParMatrix Me_petsc(&Me_mat);
         PetscErrorCode ierr = MatAYPX(dRede_petsc, -h, Me_petsc,
                                       MatStructure::DIFFERENT_NONZERO_PATTERN);
         MFEM_VERIFY(ierr == PETSC_SUCCESS, "MatAYPX(dRede) failed");

         Array<int> offsets(4);
         offsets[0] = 0;
         offsets[1] = H1tsize;
         offsets[2] = H1tsize;
         offsets[3] = L2tsize;
         offsets.PartialSum();

         BlockOperator block_op(offsets);
         block_op.SetBlock(0, 0, &dRxdx_petsc);
         block_op.SetBlock(0, 1, &dRxdv_petsc);
         block_op.SetBlock(1, 0, &dRvdx_petsc);
         block_op.SetBlock(1, 1, &dRvdv_petsc);
         block_op.SetBlock(1, 2, &dRvde_petsc);
         block_op.SetBlock(2, 0, &dRedx_petsc);
         block_op.SetBlock(2, 1, &dRedv_petsc);
         block_op.SetBlock(2, 2, &dRede_petsc);

         // Recreate the PETSc matrix every time. Trying to update the
         // underlying Mat in-place is error-prone because the temporary
         // block/matrix objects go out of scope after this call.
         petsc_jacobian.reset(
            new PetscParMatrix(comm, &block_op, Operator::PETSC_MATAIJ));
         return *petsc_jacobian;
#else
         MFEM_ABORT("MFEM is not built with PETSc");
#endif
      }

      return *jacobian;
   }
}

LagrangianHydroOperator::LagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   const IntegrationRule &ir,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   std::shared_ptr<DifferentiableOperator> update_qdata,
   std::shared_ptr<DifferentiableOperator> dtest_mf,
   std::shared_ptr<DifferentiableOperator> momentum_mf,
   std::shared_ptr<DifferentiableOperator> momentum_pa,
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf,
   std::shared_ptr<DifferentiableOperator> energy_conservation_pa,
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf,
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf,
   std::shared_ptr<DifferentiableOperator> density_mf,
   std::shared_ptr<DifferentiableOperator> taylor_source_mf,
   std::shared_ptr<QuadratureData> qdata,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type,
   Vector& external_data,
   const bool use_petsc) :
   TimeDependentOperator(2*H1.GetVSize()+L2.GetVSize()),
   H1(H1),
   L2(L2),
   H1c(H1.GetParMesh(), H1.FEColl(), 1),
   ess_tdof(ess_tdof),
   ir(ir),
   x0(x0_gf),
   rho0(rho0_gf),
   material(material_gf),
   update_qdata(update_qdata),
   dtest_mf(dtest_mf),
   momentum_mf(momentum_mf),
   momentum_pa(momentum_pa),
   energy_conservation_mf(energy_conservation_mf),
   energy_conservation_pa(energy_conservation_pa),
   total_internal_energy_mf(total_internal_energy_mf),
   total_kinetic_energy_mf(total_kinetic_energy_mf),
   density_mf(density_mf),
   taylor_source_mf(taylor_source_mf),
   qdata(qdata),
   mesh_nodes(&H1),
   rhsvc(&H1c),
   dvc(&H1c),
   Mv_blf(&H1),
   Me_blf(&L2),
   rho0_coeff(rho0_coeff),
   RHSv(H1.GetTrueVSize()),
   rhsv(H1.GetVSize()),
   X(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
   Xv(H1.GetTrueVSize()),
   Xvc(H1c.GetTrueVSize()),
   Xe(L2.GetTrueVSize()),
   K(2*H1.GetTrueVSize()+L2.GetTrueVSize()),
   B(H1c.GetTrueVSize()),
   RHSe(L2.GetTrueVSize()),
   rhse(L2.GetVSize()),
   nl2dofs(L2.GetFE(0)->GetDof()),
   fd_gradient(fd_gradient),
   dump_jacobians(dump_jacobians),
   nonlinear_maximum_iterations(nonlinear_maximum_iterations),
   nonlinear_relative_tolerance(nonlinear_relative_tolerance),
   krylov_maximum_iterations(krylov_maximum_iterations),
   preconditioner_lag(preconditioner_lag),
   preconditioner_type(preconditioner_type),
   external_data(external_data),
   use_petsc(use_petsc)
{
   Mv = new MassPAOperator(H1c, ir, rho0_coeff);
   Array<int> empty_tdofs;
   Mv_Jprec = new OperatorJacobiSmoother(Mv->GetBF(), empty_tdofs);

   Me = new MassPAOperator(L2, ir, rho0_coeff);

   // Inside the above constructors for mass, there is reordering of the mesh
   // nodes which is performed on the host. Since the mesh nodes are a
   // subvector, so we need to sync with the rest of the base vector (which
   // is assumed to be in the memory space used by the mfem::Device).
   H1.GetParMesh()->GetNodes()->ReadWrite();
   // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
   // we must enforce v_x/y/z = 0 for the velocity components.
   const int bdr_attr_max = H1.GetMesh()->bdr_attributes.Max();
   Array<int> ess_bdr(bdr_attr_max);
   for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
   {
      ess_bdr = 0;
      ess_bdr[c] = 1;
      H1c.GetEssentialTrueDofs(ess_bdr, c_tdofs[c]);
      c_tdofs[c].Read();
   }

   Mv_blf.AddDomainIntegrator(new VectorMassIntegrator(rho0_coeff, &ir));
   Mv_blf.Assemble();
   Mv_blf.FormSystemMatrix(mfem::Array<int>(), Mv_mat);

   Me_blf.AddDomainIntegrator(new MassIntegrator(rho0_coeff, &ir));
   Me_blf.Assemble();
   Me_blf.FormSystemMatrix(mfem::Array<int>(), Me_mat);

   residual.reset(new LagrangianHydroResidualOperator(*this, X, fd_gradient,
                                                      dump_jacobians));

   if (use_petsc)
   {
#ifdef MFEM_USE_PETSC
      snes = new PetscNonlinearSolver(MPI_COMM_WORLD);
      snes->SetOperator(*residual);
      snes->SetRelTol(nonlinear_relative_tolerance);
      snes->SetMaxIter(nonlinear_maximum_iterations);
      snes->SetJacobianType(Operator::PETSC_MATAIJ);
#else
      MFEM_ABORT("MFEM is not built with PETSc");
#endif
   }
   else
   {
      preconditioner.reset(new Preconditioner(*this));

      auto gmres = new GMRESSolver(MPI_COMM_WORLD);
      gmres->SetKDim(500);
      gmres->SetMaxIter(krylov_maximum_iterations);
      gmres->SetRelTol(1e-12);
      gmres->SetAbsTol(0.0);
      //gmres->SetPrintLevel(IterativeSolver::PrintLevel().Summary());
      gmres->SetPrintLevel(IterativeSolver::PrintLevel().None());
      gmres->SetPreconditioner(*preconditioner);
      krylov.reset(gmres);

      newton.reset(new LineSearchNewtonSolver(MPI_COMM_WORLD, *this));
      newton->SetPrintLevel(IterativeSolver::PrintLevel().Summary());
      //newton->SetPrintLevel(IterativeSolver::PrintLevel().None());
      newton->SetOperator(*residual);
      newton->SetSolver(*krylov);
      newton->SetMaxIter(nonlinear_maximum_iterations);
      newton->SetRelTol(nonlinear_relative_tolerance);
      newton->SetAbsTol(0.0);
      // newton->SetAdaptiveLinRtol();
   }
}

void LagrangianHydroOperator::Mult(const Vector &S, Vector &dSdt) const
{
   UpdateMesh(S);
   UpdateQuadratureData(S);

   auto sptr = const_cast<Vector*>(&S);
   const int H1vsize = H1.GetVSize();

   ParGridFunction x, v, e;
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1vsize);
   e.MakeRef(&L2, *sptr, 2*H1vsize);

   ParGridFunction dx, dv, de;
   dx.MakeRef(&H1, dSdt, 0);
   dv.MakeRef(&H1, dSdt, H1vsize);
   de.MakeRef(&L2, dSdt, 2*H1vsize);

   // solve position
   dx = v;

   // out << ">>> dx\n";
   // pretty_print(dx);

   // solve velocity
   {
      dv = 0.0;

      MultiVector Y{RHSv};
      momentum_pa->Mult(MultiVector{qdata->stressp}, Y);
      RHSv.Neg();
      H1.GetRestrictionMatrix()->MultTranspose(RHSv, rhsv);

      // pretty_print(rhsv);

      // solve for each velocity component
      const int size = H1c.GetVSize();
      const Operator *Pconf = H1c.GetProlongationMatrix();
      for (int c = 0; c < H1.GetMesh()->Dimension(); c++)
      {
         dvc.MakeRef(&H1c, dSdt, H1vsize + c*size);
         rhsvc.MakeRef(&H1c, rhsv, c*size);
         if (Pconf)
         {
            Pconf->MultTranspose(rhsvc, B);
         }
         else
         {
            B = rhsvc;
         }

         CGSolver cg(H1c.GetParMesh()->GetComm());
         cg.SetPreconditioner(*Mv_Jprec);
         cg.SetOperator(*Mv);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(300);
         cg.SetPrintLevel(-1);

         H1c.GetRestrictionMatrix()->Mult(dvc, Xvc);
         Mv->SetEssentialTrueDofs(c_tdofs[c]);
         Mv->EliminateRHS(B);
         cg.Mult(B, Xvc);
         if (Pconf)
         {
            Pconf->Mult(Xvc, dvc);
         }
         else
         {
            dvc = Xvc;
         }
         dvc.GetMemory().SyncAlias(dSdt.GetMemory(), dvc.Size());
      }
   }
   // out << ">>> dv\n";
   // pretty_print(dv);

   // solve energy
   {
      de = 0.0;

      MultiVector Y{RHSe};
      energy_conservation_pa->Mult(MultiVector{v, qdata->stressp}, Y);
      L2.GetRestrictionMatrix()->MultTranspose(RHSe, rhse);

      if (problem == 0 && L2.GetMesh()->Dimension() == 2)
      {
         LinearForm e_source(&L2);
         e_source = 0.0;
         L2.GetMesh()->DeleteGeometricFactors();

         FunctionCoefficient coeff(taylor_source);
         DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &ir);
         e_source.AddDomainIntegrator(d);
         e_source.UseFastAssembly(true);
         e_source.Assemble();

         rhse += e_source;
      }

      CGSolver cg(L2.GetParMesh()->GetComm());
      cg.SetOperator(*Me);
      cg.iterative_mode = false;
      cg.SetRelTol(1e-8);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(300);
      cg.SetPrintLevel(-1);
      cg.Mult(rhse, de);
      de.GetMemory().SyncAlias(dSdt.GetMemory(), de.Size());

      // out << ">>> de\n";
      // pretty_print(de);
   }

   qdata_is_current = false;

   // out << ">>> dSdt\n";
   // pretty_print(dSdt);
}

void LagrangianHydroOperator::ImplicitSolve(const real_t dt, const Vector &x,
                                            Vector &k)
{
   auto xptr = const_cast<Vector*>(&x);

   Vector xx, xv, xe;
   xx.MakeRef(*xptr, 0, H1.GetVSize());
   xv.MakeRef(*xptr, H1.GetVSize(), H1.GetVSize());
   xe.MakeRef(*xptr, 2*H1.GetVSize(), L2.GetVSize());

   Xx.MakeRef(X, 0, H1.GetTrueVSize());
   Xv.MakeRef(X, H1.GetTrueVSize(), H1.GetTrueVSize());
   Xe.MakeRef(X, 2*H1.GetTrueVSize(), L2.GetTrueVSize());

   H1.GetRestrictionMatrix()->Mult(xx, Xx);
   H1.GetRestrictionMatrix()->Mult(xv, Xv);
   L2.GetRestrictionMatrix()->Mult(xe, Xe);

   Xx.SyncAliasMemory(X);
   Xv.SyncAliasMemory(X);
   Xe.SyncAliasMemory(X);

   residual->SetTimeStep(dt);

   Vector zero;
   K = X;
   if (use_petsc)
   {
#ifdef MFEM_USE_PETSC
      // PETSc's SNES may cache internal state that depends on the operator.
      // Recreate the SNES object periodically (and when the timestep
      // changes) to avoid stale Jacobian/preconditioner state across
      // time steps/stages, especially when users request Jacobian lagging
      // through `-snes_lag_jacobian`.
      if (current_dt != dt || snes == nullptr || petsc_lag >= petsc_lag_steps)
      {
         current_dt = dt;
         petsc_lag = 0;
         delete snes;
         snes = new PetscNonlinearSolver(MPI_COMM_WORLD);
         snes->SetOperator(*residual);
         snes->SetRelTol(nonlinear_relative_tolerance);
         snes->SetMaxIter(nonlinear_maximum_iterations);
         snes->SetJacobianType(Operator::PETSC_MATAIJ);
      }
      petsc_lag++;

      snes->Mult(zero, K);
      newton_converged = (snes->GetConverged() > 0);
#else
      MFEM_ABORT("MFEM is not built with PETSc");
#endif
   }
   else
   {
      if (current_dt != dt || lag >= preconditioner_lag)
      {
         lag = 0;
         current_dt = dt;
         preconditioner->SetRebuildFlag(true);
      }
      lag++;

      newton->Mult(zero, K);

      // If some SDIRK stage didn't converge, keep it at false.
      // (these are reset after every time step).
      if (newton_converged)
      {
         newton_converged = newton->GetConverged();
      }
      newton_max_iter = std::max(newton_max_iter, newton->GetNumIterations());
   }

   Kx.MakeRef(K, 0, H1.GetTrueVSize());
   Kv.MakeRef(K, H1.GetTrueVSize(), H1.GetTrueVSize());
   Ke.MakeRef(K, 2*H1.GetTrueVSize(), L2.GetTrueVSize());

   Vector kx, kv, ke;
   kx.MakeRef(k, 0, H1.GetVSize());
   kv.MakeRef(k, H1.GetVSize(), H1.GetVSize());
   ke.MakeRef(k, 2*H1.GetVSize(), L2.GetVSize());

   H1.GetProlongationMatrix()->Mult(Kx, kx);
   H1.GetProlongationMatrix()->Mult(Kv, kv);
   L2.GetProlongationMatrix()->Mult(Ke, ke);
}

void LagrangianHydroOperator::UpdateMesh(const Vector &S) const
{
   Vector* sptr = const_cast<Vector*>(&S);
   mesh_nodes.MakeRef(&H1, *sptr, 0);
   H1.GetParMesh()->NewNodes(mesh_nodes, false);
}

void LagrangianHydroOperator::ResetTimeStepEstimate()
{
   external_data[EXT_DATA_IDX::DT_ESTIMATE] =
      std::numeric_limits<double>::infinity();
   dt_est_stages = std::numeric_limits<real_t>::infinity();
}

void LagrangianHydroOperator::ResetQuadratureData() const
{
   qdata_is_current = false;
}

void LagrangianHydroOperator::ResetNewtonInfo()
{
   newton_converged = true;
   newton_max_iter = -1;
}

real_t LagrangianHydroOperator::GetTimeStepEstimate(const Vector &S)
{
   MFEM_VERIFY(dtest_mf, "dt estimate operator (dtest_mf) is null");
   MFEM_VERIFY(qdata, "quadrature data (qdata) is null");

   UpdateMesh(S);
   UpdateQuadratureData(S);

   real_t dt_est_local = LocalTimeStepEstimate(S);
   if (stage_dt_estimate) { dt_est_local = fmin(dt_est_local, dt_est_stages); }

   real_t dt_est_global;
   MPI_Allreduce(&dt_est_local, &dt_est_global, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN,
                 L2.GetComm());

   return dt_est_global;
}

real_t LagrangianHydroOperator::InternalEnergy(ParGridFunction &e)
{
   const auto mt = Device::GetDeviceMemoryType();
   Vector E(L2.GetTrueVSize(), mt), Y(L2.GetTrueVSize(), mt);
   L2.GetRestrictionMatrix()->Mult(e, E);
   MultiVector MY{Y};
   total_internal_energy_mf->Mult(MultiVector{E, rho0, x0}, MY);
   const real_t ie_local = Y.Sum();
   real_t ie_global = 0.0;
   MPI_Allreduce(&ie_local, &ie_global, 1, MPI_DOUBLE, MPI_SUM,
                 L2.GetParMesh()->GetComm());
   return ie_global;
}

real_t LagrangianHydroOperator::KineticEnergy(ParGridFunction &v)
{
   const auto mt = Device::GetDeviceMemoryType();
   Vector Y(L2.GetTrueVSize(), mt);
   MultiVector MY{Y};
   total_kinetic_energy_mf->Mult(MultiVector{v, rho0, x0}, MY);
   const real_t ke_local = Y.Sum();
   real_t ke_global = 0.0;
   MPI_Allreduce(&ke_local, &ke_global, 1, MPI_DOUBLE, MPI_SUM,
                 H1.GetParMesh()->GetComm());
   return ke_global;
}

void LagrangianHydroOperator::ComputeDensity(ParGridFunction &rho)
{
   rho.SetSpace(&L2);

   ParGridFunction rhs_l(&L2);

   Vector rho0_t(L2.GetTrueVSize()),
          rho_t(L2.GetTrueVSize()),
          rhs(L2.GetTrueVSize());

   const int l2dofs_cnt = L2.GetFE(0)->GetDof();
   DenseMatrix Mrho(l2dofs_cnt);
   DenseMatrixInverse inv(&Mrho);
   Vector rhs_e(l2dofs_cnt), rho_z(l2dofs_cnt);
   Array<int> dofs(l2dofs_cnt);
   MassIntegrator mi(&ir);

   L2.GetProlongationMatrix()->MultTranspose(rho0, rho0_t);
   MultiVector Y{rhs};
   density_mf->Mult(MultiVector{rho0_t, x0}, Y);
   L2.GetProlongationMatrix()->Mult(rhs, rhs_l);

   for (int e = 0; e < L2.GetParMesh()->GetNE(); e++)
   {
      const FiniteElement &fe = *L2.GetFE(e);
      ElementTransformation &eltr = *L2.GetElementTransformation(e);
      L2.GetElementDofs(e, dofs);
      mi.AssembleElementMatrix(fe, eltr, Mrho);
      inv.Factor();
      rhs_l.GetElementDofValues(e, rhs_e);
      inv.Mult(rhs_e, rho_z);
      rho.SetSubVector(dofs, rho_z);
   }
}

void LagrangianHydroOperator::DensityScatter()
{
   Mesh m0 = H1.GetParMesh()->GetSerialMesh(0);
   GridFunction r0 = rho0.GetSerialGridFunction(0, m0);

   if (Mpi::WorldRank() != 0) { return; }

   std::ofstream fstream_rho;
   fstream_rho.open("rho.out");
   fstream_rho.precision(8);

   const int nqp = ir.GetNPoints();
   Vector pos(2);
   const int NE = m0.GetNE();
   for (int e = 0; e < NE; e++)
   {
      ElementTransformation &Tr = *m0.GetElementTransformation(e);

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, pos);

         double detJ = Tr.Weight();
         // rho0 * detJ0 / detJ.
         double rho = r0.GetValue(Tr, ip)  * (1.0 / NE) / detJ;
         double r = sqrt(pos(0) * pos(0) + pos(1) * pos(1));

         fstream_rho << r << " " << rho << "\n";
      }
   }

   fstream_rho.close();
}

real_t LagrangianHydroOperator::ComputeMinDet(const Vector &S) const
{
   auto sptr = const_cast<Vector*>(&S);
   Vector x, x_loc(H1.GetVSize());
   x.MakeRef(*sptr, 0, H1.GetTrueVSize());
   H1.GetProlongationMatrix()->Mult(x, x_loc);

   ParGridFunction xgf;
   xgf.MakeRef(&H1, x_loc, 0);

   auto g = GeometricFactors(xgf, ir, GeometricFactors::JACOBIANS |
                             GeometricFactors::DETERMINANTS);
   auto min = g.detJ.Min();
   MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, L2.GetComm());

   return min;
}

void LagrangianHydroOperator::UpdateQuadratureData(const Vector &S) const
{
   if (qdata_is_current) { return; }
   // out << "updating qdata\n";
   qdata_is_current = true;

   auto sptr = const_cast<Vector*>(&S);
   const int H1vsize = H1.GetVSize();
   ParGridFunction x, v, e;
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1vsize);
   e.MakeRef(&L2, *sptr, 2*H1vsize);
   MultiVector Y{qdata->stressp};
   update_qdata->Mult(MultiVector{v, rho0, x0, x, material, e}, Y);

   // As stock Laghos: the estimate is the min over all the updates
   if (stage_dt_estimate)
   {
      dt_est_stages = fmin(dt_est_stages, LocalTimeStepEstimate(S));
   }
}

real_t LagrangianHydroOperator::LocalTimeStepEstimate(const Vector &S) const
{
   auto sptr = const_cast<Vector*>(&S);
   const int H1vsize = H1.GetVSize();
   ParGridFunction x, v, e;
   x.MakeRef(&H1, *sptr, 0);
   v.MakeRef(&H1, *sptr, H1vsize);
   e.MakeRef(&L2, *sptr, 2*H1vsize);
   MultiVector Y{qdata->dt_est};
   dtest_mf->Mult(MultiVector{v, rho0, x0, x, material, e}, Y);
   return qdata->dt_est.Min();
}

MemoryClass LagrangianHydroOperator::GetMemoryClass() const
{
   return Device::GetDeviceMemoryClass();
}

LagrangianHydroOperator::~LagrangianHydroOperator()
{
#ifdef MFEM_USE_PETSC
   delete snes;
#endif
   delete Mv;
   delete Mv_Jprec;
   delete Me;
}

template <int DIM>
LagrangianHydroOperator *CreateLagrangianHydroOperator(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type,
   const bool use_petsc)
{
   ParMesh &mesh = *H1.GetParMesh();

   auto qdata = std::make_shared<QuadratureData>(mesh, ir);

   int ne_loc = mesh.GetNE(), ne_global = 0;
   real_t vol_loc = 0.0, vol_global = 0.0;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      vol_loc += mesh.GetElementVolume(e);
   }
   MPI_Allreduce(&vol_loc, &vol_global, 1, MPI_DOUBLE, MPI_SUM, mesh.GetComm());
   MPI_Allreduce(&ne_loc, &ne_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());

   real_t &h0 = external_data[EXT_DATA_IDX::H0];
   switch (mesh.GetElementBaseGeometry(0))
   {
      case Geometry::SEGMENT: h0 = vol_global / ne_global; break;
      case Geometry::SQUARE: h0 = sqrt(vol_global / ne_global); break;
      case Geometry::TRIANGLE: h0 = sqrt(2.0 * vol_global / ne_global); break;
      case Geometry::CUBE: h0 = pow(vol_global / ne_global, 1./3.); break;
      case Geometry::TETRAHEDRON: h0 = pow(6.0 * vol_global / ne_global,
                                              1./3.); break;
      default: MFEM_ABORT("Unknown zone type!");
   }
   h0 /= (double) H1.GetOrder(0);

   // const real_t h0 = sqrt(vol_global / ne_global) /
   //                   static_cast<real_t>(H1.GetOrder(0));

   // qdata->order_v = order_v;
   qdata->dt_est = std::numeric_limits<real_t>::infinity();

   auto d_external_data = external_data.ReadWrite();

   Array<int> all_domain_attr(mesh.attributes.Max());
   all_domain_attr = 1;

   // Contains a ParametricSpace.
   std::shared_ptr<DifferentiableOperator> dt_est_mf;
   {
      future::tuple dt_est_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      future::tuple dt_est_kernel_oo = {Identity<DT_EST>{}};

      std::vector dt_est_inputs =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      std::vector dt_est_outputs =
      {
         FieldDescriptor{DT_EST, &qdata->R}
      };

      dt_est_mf = std::make_shared<DifferentiableOperator>(
                     dt_est_inputs, dt_est_outputs, mesh);
      dt_est_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      TimeStepEstimateQFunction<DIM> dt_est_qf(d_external_data);
      dt_est_mf->AddDomainIntegrator(dt_est_qf, dt_est_kernel_ao,
                                     dt_est_kernel_oo,
                                     ir,
                                     all_domain_attr);
   }

   // Contains a ParametricSpace.
   std::shared_ptr<DifferentiableOperator> update_qdata;
   {
      future::tuple update_qdata_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      future::tuple update_qdata_kernel_oo = {Identity<STRESS_TENSOR>{}};

      std::vector<FieldDescriptor> update_qdata_inputs =
      {
         {VELOCITY, &H1},
         {DENSITY0, &L2},
         {COORDINATES0, &H1},
         {COORDINATES, &H1},
         {MATERIAL, material_gf.ParFESpace()},
         {SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      std::vector<FieldDescriptor> update_qdata_outputs =
      {
         {STRESS_TENSOR, &qdata->StressSpace}
      };

      update_qdata = std::make_shared<DifferentiableOperator>(
                        update_qdata_inputs, update_qdata_outputs, mesh);
      update_qdata->SetMultLevel(DifferentiableOperator::LVECTOR);
      UpdateQuadratureDataQFunction<DIM> update_qdata_qf(d_external_data);
      update_qdata->AddDomainIntegrator(update_qdata_qf, update_qdata_kernel_ao,
                                        update_qdata_kernel_oo,
                                        ir,
                                        all_domain_attr);
   }

   // Create momentum operator
   std::shared_ptr<DifferentiableOperator> momentum_mf;
   {
      future::tuple momentum_mf_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      future::tuple momentum_mf_kernel_oo = {Gradient<VELOCITY>{}};

      // <sigma, grad(w) * J^-T> * det(J) * weights
      // <sigma(J^-T det(J) weights), grad(w)>

      std::vector momentum_mf_inputs =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
      };

      std::vector momentum_mf_outputs =
      {
         FieldDescriptor{VELOCITY, &H1}
      };

      momentum_mf = std::make_shared<DifferentiableOperator>(
                       momentum_mf_inputs, momentum_mf_outputs, mesh);

      MomentumQFunction<DIM> momentum_qf(d_external_data);
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      momentum_mf->AddDomainIntegrator(momentum_qf, momentum_mf_kernel_ao,
                                       momentum_mf_kernel_oo, ir, all_domain_attr, derivatives);
   }

   std::shared_ptr<DifferentiableOperator> momentum_pa;
   {
      future::tuple momentum_pa_kernel_ao = {Identity<STRESS_TENSOR>{}};
      future::tuple momentum_pa_kernel_oo = {Gradient<VELOCITY>{}};

      std::vector<FieldDescriptor> momentum_pa_inputs = {{STRESS_TENSOR, &qdata->StressSpace}};
      std::vector<FieldDescriptor> momentum_pa_outputs = {{VELOCITY, &H1}};

      momentum_pa = std::make_shared<DifferentiableOperator>(
                       momentum_pa_inputs, momentum_pa_outputs, mesh);

      MomentumPAQFunction<DIM> momentum_pa_qf;
      momentum_pa->AddDomainIntegrator(momentum_pa_qf, momentum_pa_kernel_ao,
                                       momentum_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create energy conservation operator
   std::shared_ptr<DifferentiableOperator> energy_conservation_mf;
   {
      future::tuple energy_conservation_mf_kernel_ao =
      {
         Gradient<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Gradient<COORDINATES>{},
         Value<MATERIAL>{},
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Weight{}
      };

      future::tuple energy_conservation_mf_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      // <sigma, grad(v) * inv(J) * phi> * det(J) * w
      // <sigma(J^-T det(J) w), grad(v) * inv(J)>

      std::vector energy_conservation_mf_inputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1},
         FieldDescriptor{COORDINATES, &H1},
         FieldDescriptor{MATERIAL, material_gf.ParFESpace()},
      };

      std::vector energy_conservation_mf_outputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      energy_conservation_mf =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_mf_inputs, energy_conservation_mf_outputs, mesh);

      EnergyConservationQFunction<DIM> energy_conservation_qf(d_external_data);
      auto derivatives =
         std::integer_sequence<size_t, VELOCITY, COORDINATES, SPECIFIC_INTERNAL_ENERGY> {};
      energy_conservation_mf->AddDomainIntegrator(
         energy_conservation_qf, energy_conservation_mf_kernel_ao,
         energy_conservation_mf_kernel_oo, ir, all_domain_attr, derivatives);
   }

   std::shared_ptr<DifferentiableOperator> energy_conservation_pa;
   {
      future::tuple energy_conservation_pa_kernel_ao = {Gradient<VELOCITY>{}, Identity<STRESS_TENSOR>{}};
      future::tuple energy_conservation_pa_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector<FieldDescriptor> energy_conservation_pa_inputs =
      {
         {VELOCITY, &H1},
         {STRESS_TENSOR, &qdata->StressSpace}
      };

      std::vector<FieldDescriptor> energy_conservation_pa_outputs =
      {
         {SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      energy_conservation_pa =
         std::make_shared<DifferentiableOperator>(
            energy_conservation_pa_inputs, energy_conservation_pa_outputs, mesh);
      energy_conservation_pa->SetMultLevel(DifferentiableOperator::LVECTOR);

      EnergyConservationPAQFunction<DIM> energy_conservation_pa_qf;
      energy_conservation_pa->AddDomainIntegrator(
         energy_conservation_pa_qf, energy_conservation_pa_kernel_ao,
         energy_conservation_pa_kernel_oo, ir, all_domain_attr);
   }

   // Create total internal energy operator
   std::shared_ptr<DifferentiableOperator> total_internal_energy_mf;
   {
      future::tuple total_internal_energy_kernel_ao =
      {
         Value<SPECIFIC_INTERNAL_ENERGY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      future::tuple total_internal_energy_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector total_internal_energy_inputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };

      std::vector total_internal_energy_outputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      total_internal_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_internal_energy_inputs,
            total_internal_energy_outputs,
            mesh);
      total_internal_energy_mf->SetMultLevel(DifferentiableOperator::LVECTOR);

      TotalInternalEnergyQFunction<DIM> total_internal_energy_qf;
      total_internal_energy_mf->AddDomainIntegrator(
         total_internal_energy_qf, total_internal_energy_kernel_ao,
         total_internal_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create total kinetic energy operator
   std::shared_ptr<DifferentiableOperator> total_kinetic_energy_mf;
   {
      future::tuple total_kinetic_energy_kernel_ao =
      {
         Value<VELOCITY>{},
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      future::tuple total_kinetic_energy_kernel_oo = {Value<DENSITY0>{}};

      std::vector total_kinetic_energy_inputs =
      {
         FieldDescriptor{VELOCITY, &H1},
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };

      // The kinetic energy density is an L2 field
      std::vector total_kinetic_energy_outputs =
      {
         FieldDescriptor{DENSITY0, &L2}
      };

      total_kinetic_energy_mf =
         std::make_shared<DifferentiableOperator>(
            total_kinetic_energy_inputs,
            total_kinetic_energy_outputs, mesh);
      total_kinetic_energy_mf->SetMultLevel(DifferentiableOperator::LVECTOR);
      TotalKineticEnergyQFunction<DIM> total_kinetic_energy_qf;
      total_kinetic_energy_mf->AddDomainIntegrator(
         total_kinetic_energy_qf, total_kinetic_energy_kernel_ao,
         total_kinetic_energy_kernel_oo, ir, all_domain_attr);
   }

   // Create density operator
   std::shared_ptr<DifferentiableOperator> density_mf;
   {
      future::tuple density_kernel_ao =
      {
         Value<DENSITY0>{},
         Gradient<COORDINATES0>{},
         Weight{}
      };

      future::tuple density_kernel_oo = {Value<DENSITY0>{}};

      std::vector density_inputs =
      {
         FieldDescriptor{DENSITY0, &L2},
         FieldDescriptor{COORDINATES0, &H1}
      };

      std::vector density_outputs =
      {
         FieldDescriptor{DENSITY0, &L2}
      };

      density_mf = std::make_shared<DifferentiableOperator>(
                      density_inputs, density_outputs, mesh);
      density_mf->SetMultLevel(DifferentiableOperator::LVECTOR);

      DensityQFunction<DIM> density_qf;
      density_mf->AddDomainIntegrator(density_qf, density_kernel_ao,
                                      density_kernel_oo, ir, all_domain_attr);
   }

   // Create taylor source oeprator
   std::shared_ptr<DifferentiableOperator> taylor_source_mf;
   {
      future::tuple taylor_source_kernel_ao =
      {
         Value<COORDINATES>{},
         Gradient<COORDINATES>{},
         Weight{}
      };

      future::tuple taylor_source_kernel_oo = {Value<SPECIFIC_INTERNAL_ENERGY>{}};

      std::vector taylor_source_inputs =
      {
         FieldDescriptor{COORDINATES, &H1}
      };

      std::vector taylor_source_outputs =
      {
         FieldDescriptor{SPECIFIC_INTERNAL_ENERGY, &L2}
      };

      taylor_source_mf = std::make_shared<DifferentiableOperator>(
                            taylor_source_inputs, taylor_source_outputs, mesh);

      TaylorSourceQFunction<DIM> taylor_source_qf;
      auto derivatives = std::integer_sequence<size_t, COORDINATES> {};
      taylor_source_mf->AddDomainIntegrator(taylor_source_qf, taylor_source_kernel_ao,
                                            taylor_source_kernel_oo, ir, all_domain_attr,
                                            derivatives);
   }

   return new LagrangianHydroOperator(
             H1,
             L2,
             ess_tdof,
             ir,
             rho0_coeff,
             x0_gf,
             rho0_gf,
             material_gf,
             update_qdata,
             dt_est_mf,
             momentum_mf,
             momentum_pa,
             energy_conservation_mf,
             energy_conservation_pa,
             total_internal_energy_mf,
             total_kinetic_energy_mf,
             density_mf,
             taylor_source_mf,
             qdata,
             fd_gradient,
             dump_jacobians,
             nonlinear_maximum_iterations,
             nonlinear_relative_tolerance,
             krylov_maximum_iterations,
             preconditioner_lag,
             preconditioner_type,
             external_data,
             use_petsc);
}

// Explicit instantiations for the 2D and 3D meshes

template LagrangianHydroOperator *CreateLagrangianHydroOperator<2>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type,
   const bool use_petsc);

template LagrangianHydroOperator *CreateLagrangianHydroOperator<3>(
   ParFiniteElementSpace &H1,
   ParFiniteElementSpace &L2,
   Array<int> &ess_tdof,
   FunctionCoefficient &rho0_coeff,
   ParGridFunction &x0_gf,
   ParGridFunction &rho0_gf,
   ParGridFunction &material_gf,
   Vector &external_data,
   const IntegrationRule &ir,
   const bool &fd_gradient,
   const int &dump_jacobians,
   const int &nonlinear_maximum_iterations,
   const real_t &nonlinear_relative_tolerance,
   const int &krylov_maximum_iterations,
   const int &preconditioner_lag,
   const PRECONDITIONER_TYPE &preconditioner_type,
   const bool use_petsc);

} // namespace hydrodynamics

} // namespace mfem
