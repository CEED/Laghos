               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp


## Purpose

**Laghos** (LAGrangian High-Order Solver) is a miniapp that solves the
time-dependent Euler equation of compressible gas dynamics in a moving
Lagrangian frame using unstructured high-order finite element spatial
discretization and explicit high-order time-stepping.

Laghos is based on the numerical algorithm described in the following
article:

> V. Dobrev, Tz. Kolev and R. Rieben,<br>
> [High-order curvilinear finite element methods for Lagrangian hydrodynamics](https://doi.org/10.1137/120864672), <br>
> *SIAM Journal on Scientific Computing*, (34) 2012, pp.B606–B641.

Laghos captures the basic structure of many other compressible shock
hydrocodes, including the [BLAST code](http://llnl.gov/casc/blast) at
[Lawrence Livermore National Laboratory](http://llnl.gov). The miniapp
is build on top of a general discretization library, [MFEM](http://mfem.org),
separating the pointwise physics from finite element and meshing concerns.

The Laghos miniapps is part of the [CEED software suite](http://ceed.exascaleproject.org/software),
a collection of software benchmarks, miniapps, libraries and APIs for
efficient exascale discretizations based on high-order finite element
and spectral element methods. See http://github.com/ceed for more
information and source code availability.

The CEED research is supported by the [Exascale Computing Project](https://exascaleproject.org/exascale-computing-project)
(17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
organizations (Office of Science and the National Nuclear Security
Administration) responsible for the planning and preparation of a
[capable exascale ecosystem](https://exascaleproject.org/what-is-exascale),
including software, applications, hardware, advanced system engineering and early
testbed platforms, in support of the nation’s exascale computing imperative.

## Characteristics

Laghos exposes the principal computational kernels of explicit
time-dependent shock-capturing compressible flow, including the
FLOP-intensive definition of artificial viscosity at quadrature points.

It includes the following components, the majority of which are
frequently found in HPC simulation codes:

- Support for unstructured mesh, in 2D and 3D, both with quad/hex and
  triangle/tet elements. Serial and parallel mesh refinement options can
  be set with a command-line flag.

- Explicit time-stepping loop with a variety of time integrator
  options. Laghos supports explicit Runge-Kutta ODE solvers of orders 1,
  2, 3, 4 and 6.

- Continuous and discontinuous high-order finite element discretization
  spaces of runtime-specified order.

- Constant-in-time *mass matrix* that is inverted iteratively on each
  time step ("assemble" once, evaluate many times) coupled with a
  time-dependent *force matrix* that is "assembled" on each time step
  and evaluated just twice.

- [Partial assembly](http://ceed.exascaleproject.org/ceed-code) for
  efficient high-order operator evaluation.

- Domain-decomposed MPI parallelism.

- Moving (high-order) meshes.

- Optional in-situ visualization with [GLVis](http:/glvis.org) and data
  output for analysis with [VisIt](http://visit.llnl.gov).

## Code Structure

- `laghos.cpp` contains the main driver with time integration loop
  starting at line 290.

- The problem is formulated as solving a big ordinary differential
  equation (ODE) for the unknown high-order velocity, energy and mesh
  nodes (position).

- The right-hand side of the ODE is specified by the
  `LagrangianHydroOperator` defined on line 239 of `laghos.cpp` and
  implemented in files `laghos_solver.hpp` and `laghos_solver.cpp`.

- The orders of the velocity and position (continuous kinematic space)
  and the internal energy (discontinuous thermodynamic space) are given
  by the `-ov` and `-ot` input parameters respectively.

- The main computational kernels are...

## Building

- Unpack hypre, METIS, mfem
- make parallel in mfem
- cd miniapps/hydrodynamics
- make laghos

## Running

- Sedov problem:

```sh
mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8 -no-vis -pa
```

# Verification of Results

- Final energy should be a round-off distance from 49.5731419667

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
Lawrence Livermore National Laboratory. LLNL-CODE-XXXXXX. All Rights reserved.

