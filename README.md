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

It includes several computational motives, many of which are frequently found in
HPC simulation codes:

- Support for unstructured meshes, in 2D and 3D, with quadrilateral and
  hexahedral elements. (Triangular and tetrahedral elements can also be used, but
  with less efficient "full assembly".) Serial and parallel mesh refinement
  options can be set via a command-line flag.
- Explicit time-stepping loop with a variety of time integrator options. Laghos
  supports Runge-Kutta ODE solvers of orders 1, 2, 3, 4 and 6.
- Continuous and discontinuous high-order finite element discretization spaces
  of runtime-specified order.
- Constant-in-time *mass matrix* that is inverted iteratively on each time step
  ("assemble" once, evaluate many times) coupled with a time-dependent *force
  matrix* that is "assembled" on each time step and evaluated just twice.
- [Partial assembly](http://ceed.exascaleproject.org/ceed-code) for efficient
  high-order operator evaluation.
- Moving (high-order) meshes. Point-wise definition of mesh size and artificial
  viscosity coefficient.
- Domain-decomposed MPI parallelism.
- Optional in-situ visualization with [GLVis](http:/glvis.org) and data output
  for analysis with [VisIt](http://visit.llnl.gov).

## Code Structure

- The file `laghos.cpp` contains the main driver with the time integration loop
  starting around line 290.
- The problem is formulated as solving a big system of ordinary differential
  equations for the unknown (high-order) velocity, internal energy and mesh
  nodes (position).
- The right-hand side of the ODE is specified by the `LagrangianHydroOperator`
  defined around line 239 of `laghos.cpp` and implemented in files
  `laghos_solver.hpp` and `laghos_solver.cpp`.
- The orders of the velocity and position (continuous kinematic space)
  and the internal energy (discontinuous thermodynamic space) are given
  by the `-ov` and `-ot` input parameters respectively.
- The main computational kernels are the `Mult*` functions of the classes
  `MassPAOperator` and `ForcePAOperator` implemented in file
  `laghos_solver.cpp`. Some of these functions have specific versions for
  quadrilateral and hexahedral elements.

## Building

Laghos has the following external dependencies:

- *hypre*, used for parallel linear algebra, we recommend version 2.10.0b<br>
   https://computation.llnl.gov/casc/hypre/software.html, 

-  METIS, used for parallel domain decomposition (optional), we recommend [version 4.0.3](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz) <br>
   http://glaros.dtc.umn.edu/gkhome/metis/metis/download

- MFEM, used for (high-order) finite element discretization, we recommend version 3.4 <br>
  http://mfem.org/download
  
To build the miniapp, first download *hypre*, METIS and MFEM from the links above
and put everything on the same level as Laghos:
```sh
~> ls
Laghos/ hypre-2.10.0b.tar.gz   metis-4.0.tar.gz   mfem-3.4.tgz
```

Build hypre:
```sh
~> tar -zxvf hypre-2.10.0b.tar.gz
~> cd hypre-2.10.0b/src/
~/hypre-2.10.0b/src> ./configure --disable-fortran
~/hypre-2.10.0b/src> make -j
~/hypre-2.10.0b/src> cd ../..
```

Build metis:
```sh
~> tar -zxvf metis-4.0.3.tar.gz
~> cd metis-4.0.3
~/metis-4.0.3> make
~/metis-4.0.3> cd ..
~> ln -s metis-4.0.3 metis-4.0
```

Build the parallel version of MFEM:
```sh
~> tar -zxvf mfem-3.4.tgz
~> cd mfem-3.4/
~/mfem-3.4> make parallel -j
~> cd ..
~> ln -s mfem-3.4 mfem
```

Build Laghos
```sh
~> cd Laghos/
~> make
```

For more details, see the [MFEM building page](http://mfem.org/building/).

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

