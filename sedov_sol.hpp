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

#ifndef LAGHOS_SEDOV_SOL_HPP
#define LAGHOS_SEDOV_SOL_HPP

/// Taylor-von Neumann-Sedov blast wave solution
struct SedovSol {
  /// 1 for plane wave, 2 for cylinder, 3 for sphere
  int dim;
  /// time to compute the solution at
  double t = 0;
  /// ideal gas gamma
  double gamma;
  /// initial density = rho_0 * pow(r, -omega)
  double rho_0;
  double omega;
  /// initial blast energy
  double blast_energy;

  /// currently only supports uniform initial density
  /// computed quantities used for computing the solution
  /// these values don't depend on time
  double a;
  double b;
  double c;
  double d;
  double e;
  
  double alpha0;
  double alpha1;
  double alpha2;
  double alpha3;
  double alpha4;
  double alpha5;
  
  double V0;
  double Vv;
  double V2;
  double Vs;

  double alpha;

  /// these values depend on time
  /// shock position
  double r2;
  /// shock speed
  double U;
  /// pre-shock density
  double rho1;
  /// post-shock state
  double rho2;
  double v2;
  double p2;

  void SetTime(double t);

  void EvalSol(double r, double &rho, double &v, double &P) const;

  SedovSol(int dim, double gamma, double rho_0, double blast_energy,
           double omega = 0);
};

#endif
