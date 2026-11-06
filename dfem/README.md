               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp

                      ∂FEM version

## Overview
This directory contains the ∂FEM version of **Laghos** (**LAG**rangian
**H**igh-**O**rder **S**olver):
- [**Explicit**](https://doi.org/10.1137/120864672) and
  [**implicit**](https://arxiv.org/abs/2609.16424) time integration
- Local and global q-functions
- **CPU** and **GPU** execution
- A test suite covering the Sedov, Taylor-Green and triple-point problems

For more details about Laghos and the CEED software suite, see the
[README file](../README.md) in the top-level directory.

## Differences from the official benchmark version
1. The hydrodynamics operators use MFEM's differentiable finite element
   (∂FEM) framework, including Jacobian kernels for implicit integration.
2. Local and global q-function backends are selected at runtime with `-qf`.
3. **Explicit** and **implicit** ODE solvers are available, including RK2 SSP,
   RK3 SSP, RK4, backward Euler and implicit midpoint.
4. The executable is named `laghos_dfem`. Its command-line options differ
   from the top-level version: use `-ov` and `-oe` for the velocity and
   internal energy orders, and `-glvis` for GLVis visualization.

## Limitations
- Supported problems, in 2D and 3D:
  - `-p 0`: Taylor-Green vortex
  - `-p 1`: Sedov blast
  - `-p 3`: triple point
- Unsupported problems:
  - `-p 2`: 1D Sod shock tube
  - `-p 4`: Gresho vortex
  - `-p 5`: 2D Riemann problem, config. 12
  - `-p 6`: 2D Riemann problem, config. 6
  - `-p 7`: 2D Rayleigh-Taylor instability

## Building

The ∂FEM version requires MPI, hypre and an MFEM build from the `dfem-dev`
branch, which provides the required support and the `local` and `global`
q-function backends.
See the top-level [building instructions](../README.md#building) for the
other dependencies and **CPU** and **GPU** (CUDA or HIP) configurations.

### Makefile

From this directory, point `MFEM_DIR` to an MFEM source, build or
installation directory built from the `dfem-dev` branch:

```sh
  make -j 4 MFEM_DIR=/path/to/mfem
  make test MFEM_DIR=/path/to/mfem
```

The default build produces `laghos_dfem` and `laghos_tests`. The test executable
requires MFEM's `tests/unit/catch.hpp`; set `CATCH_DIR` to its directory if
necessary. `make test` runs a short Taylor-Green case on four MPI ranks.
See `make help` for configuration, testing and installation options.

### CMake

Configure from the top-level directory, not from `dfem/`. For an installed
MFEM from the `dfem-dev` branch and hypre, with the MFEM source tree
available for the test header:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMFEM_DIR=/path/to/mfem/lib/cmake/mfem \
  -DHYPRE_DIR=/path/to/hypre \
  -DMFEM_SOURCE_DIR=/path/to/mfem/source
cmake --build build --target laghos_dfem laghos_tests -j 4
```

The executables are placed in `build/dfem/`. If MFEM's `tests/unit/catch.hpp`
cannot be found, CMake warns and omits `laghos_tests` and its tests.
CMake uses `MPIEXEC_EXECUTABLE`, `MPIEXEC_NUMPROC_FLAG`, `MPIEXEC_PREFLAGS`
and `MPIEXEC_POSTFLAGS` to launch the tests.

## Running

Run the following examples from this directory after a Makefile build.
For a CMake build, replace `./laghos_dfem` with `../build/dfem/laghos_dfem`.
The mesh file must be readable even for problems that generate their mesh.

### Sedov blast

**Explicit** RK3 SSP in 2D, with smooth artificial viscosity:

```sh
mpirun -np 2 ./laghos_dfem -m ../data/square01_quad.mesh \
  -p 1 -dim 2 -rs 3 -ov 2 -oe 1 -s 3 -cfl 0.5 \
  -av -av-type 7 -tf 0.05 -no-save
```

**Implicit** midpoint in 2D:

```sh
mpirun -np 2 ./laghos_dfem -m ../data/square01_quad.mesh \
  -p 1 -dim 2 -rs 2 -ov 2 -oe 1 -s 12 -cfl 8 \
  -av -av-type 7 -tf 0.1 -nmi 50 -nrt 1e-10 -kmi 20 -pt 0 -no-save
```

### Taylor-Green vortex

```sh
mpirun -np 2 ./laghos_dfem -m ../data/cube01_hex.mesh \
  -p 0 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 -tf 0.05 -no-save
```

### Triple-point problem

```sh
mpirun -np 2 ./laghos_dfem -m ../data/box01_hex.mesh \
  -p 3 -dim 3 -rs 0 -ov 2 -oe 1 -s 4 -cfl 0.5 \
  -av -av-type 7 -tf 0.2 -no-save
```

Useful options include:

- `-qf local` (default) or `-qf global`: select the q-function backend.
  Internally these are `LOCAL_QF=0` and `GLOBAL_QF=1`. The tests check
  both backends against the same references.
- `-d cpu` (default), `-d debug`, `-d cuda` or `-d hip`: select an MFEM
  device supported by the build. `-dev` selects the GPU device number.
- `-s 2`, `-s 3`, `-s 4`, `-s 11`, `-s 12`, `-s 14`: RK2 SSP, RK3 SSP,
  RK4, backward Euler, implicit midpoint and SDIRK34. RK4 is the default.
- `-av -av-type 1`: stock Laghos tensor viscosity; `-av -av-type 7`:
  smooth viscosity. `-no-av` disables viscosity (the default).
- `-dt-min`: use the minimum time step estimate over the RK stages and the
  new state for explicit integration. This is off by default.
- `-glvis`: stream to GLVis; `-paraview`: write ParaView output.
  Both are off by default. Final mesh and field output is on by default;
  use `-no-save` to disable it.

## Verification of Results

The reference values and tolerances are stored in `laghos_tests.cpp`.
Both `chk` and `run` cases, **explicit** and **implicit**, check the
accepted step count, final time step, total energy, and position,
velocity and internal energy norms against those hardcoded values.
The `chk` cases are short dFEM runs of the Taylor-Green and Sedov
problems in the top-level README (entries 1-4). Their final times are
0.05, 0.01, 0.05 and 0.02, and the 3D Sedov check does not use `-E0 2`.
They are not a substitute for running the full suite.

From the top-level directory after a CMake build:

```sh
ctest --test-dir build -N -R '^dfem_'
ctest --test-dir build --output-on-failure -R '^dfem_' -j 4
```

With `LAGHOS_TESTS_GROUPS=OFF`, individual tests use the naming convention:

```text
dfem_<cpu|dbg|gpu>_<loc|glb>_<exp|imp>_<chk|run>_<pb>_<dim>_
     <q..>_<stepper>_<av>_<dtm|dtn>_np<N>
```

The name is a single line. Problem tags are `sdv`, `tgv` and `tpt`; `q21`,
for example, means Q2-Q1. Stepper tags include `rk2`, `rk3`, `rk4`, `mid`
and `bwe`. `av0` denotes no viscosity, `av1` and `av7` select viscosity
variants, and `dtm`/`dtn` denote `-dt-min` enabled/disabled.
**CPU** and debug tests run on one, two and three ranks; **GPU** tests run
on one and two ranks, using both q-function backends and the same references.

The Makefile also provides a full suite and a restricted check run:

```sh
make tests MFEM_DIR=/path/to/mfem
make tests MFEM_DIR=/path/to/mfem TESTS='[chk]' RANKS=1 DEVICES=cpu QFS=loc
```

## References

The **explicit** discretization follows the original Laghos method:

> V. Dobrev, Tz. Kolev and R. Rieben <br>
> [High-order curvilinear finite element methods for Lagrangian hydrodynamics](https://doi.org/10.1137/120864672) <br>
> *SIAM Journal on Scientific Computing*, (34) 2012, pp. B606–B641.

The **implicit** solvers follow the recent extension:

> J. Andrej, J. Camier, V. Dobrev, Tz. Kolev, B. Lazarov, K. Mittal,
> R. Rieben, B. Talamini and V. Z. Tomov <br>
> [Implicit Lagrangian Hydrodynamics with High-Order Finite Elements](https://arxiv.org/abs/2609.16424) <br>
> arXiv:2609.16424, 2026.

## Contact

You can reach the Laghos team by emailing <laghos@llnl.gov> or by leaving a
comment in the [issue tracker](https://github.com/CEED/Laghos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE in the top-level directory for details.
