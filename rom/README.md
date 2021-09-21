          _                 _               _____   ____  __  __
         | |               | |             |  __ \ / __ \|  \/  |
         | |     __ _  __ _| |__   ___  ___| |__) | |  | | \  / |
         | |    / _` |/ _` | '_ \ / _ \/ __|  _  /| |  | | |\/| |
         | |___| (_| | (_| | | | | (_) \__ \ | \ \| |__| | |  | |
         |______\__,_|\__, |_| |_|\___/|___/_|  \_\\____/|_|  |_|
                      __/ |                                     
                     |___/
 

        Reduced Order Model for High-order Lagrangian Hydrodynamics Miniapp

[![Build Status](https://travis-ci.org/CEED/Laghos.svg?branch=master)](https://travis-ci.org/CEED/Laghos)

## Purpose

**LaghosROM** (LAGrangian High-Order Solver Reduced Order Model) is a miniapp
that accelerates the time-dependent Euler equations of compressible gas
dynamics in a moving Lagrangian frame using unstructured high-order finite
element spatial discretization and explicit high-order time-stepping.

The main version of LaghosROM is in the Laghos/rom subdirectory. The purpose of
LaghosROM is to demonstrate the efficiency and accuracy of reduced order
modeling (ROM) with hyperreduction for hydrodynamics in LaghosROM. Various
options are available for the user to apply to several model problems. In
particular, time-windowing is supported, to keep reduced basis dimensions small.
Parametric ROM capabilities allow for building ROM bases from offline training
simulations using multiple PDE parameter samples.

LaghosROM is based on the following article:

> D. Copeland, S.W. Cheung, K. Huynh, and Y. Choi <br>
> [Reduced order models for Lagrangian hydrodynamics](https://arxiv.org/abs/2104.11404) <br>
> *submitted for publication*, 2021.

To see the purpose of **Laghos**, please see README.md in Laghos directory.

## Characteristics

LaghosROM with hyperreduction uses the same code as the full-order version on a
sample mesh constructed to contain sampled degrees of freedom. Consequently,
many of the full-order Laghos features can also be used in LaghosROM, e.g.
unstructured meshes, various time integrators, high-order finite element spaces,
and partial assembly.

One additional feature in LaghosROM is the option to use Gram-Schmidt to
orthogonalize the ROM bases for velocity and energy with respect to the
corresponding mass matrices. This reduces the mass matrices to identity,
obviating the need to compute the action of the mass matrix inverses.

Some features not yet available in LaghosROM are parallel computation (the
online ROM simulation is small and runs on only one MPI process) and GPU
acceleration. Support for these is planned as future work.

To see the characteristics of **Laghos**, please see README.md in the Laghos
directory.

## Code Structure

- The file `laghos.cpp` contains the main driver with the time integration loop
  for an offline (full-order) or online (ROM) simulation.
- In each time step, the ODE system of interest is constructed and solved by
  the class `LagrangianHydroOperator`, in the offline (full-order) case. For the
  online ROM case, the reduced system is constructed and solved using the
  ROM_Basis and ROM_Operator classes in `laghos_rom.hpp` and `laghos_rom.cpp`.
- In the offline case, the class ROM_Sampler samples the solution and source,
  generating ROM bases.
- All quadrature-based computations are performed in the function
  `LagrangianHydroOperator::UpdateQuadratureData` in `laghos_solver.cpp`.
- Depending on the chosen option (`-pa` for partial assembly or `-fa` for full
  assembly), the function `LagrangianHydroOperator::Mult` uses the corresponding
  method to construct and solve the final ODE system.
- The full assembly computations for all mass matrices are performed by the MFEM
  library, e.g., classes `MassIntegrator` and `VectorMassIntegrator`.  Full
  assembly of the ODE's right hand side is performed by utilizing the class
  `ForceIntegrator` defined in `laghos_assembly.hpp`.
- The partial assembly computations are performed by the classes
  `ForcePAOperator` and `MassPAOperator` defined in `laghos_assembly.hpp`.
- When partial assembly is used, the main computational kernels are the
  `Mult*` functions of the classes `MassPAOperator` and `ForcePAOperator`
  implemented in file `laghos_assembly.cpp`. These functions have specific
  versions for quadrilateral and hexahedral elements.
- The orders of the velocity and position (continuous kinematic space)
  and the internal energy (discontinuous thermodynamic space) are given
  by the `-ok` and `-ot` input parameters, respectively.

## Building

To build the dependencies of LaghosROM: 
```sh
~> cd Laghos/rom
~/Laghos/rom> make setup
```

(Optional) Clone and build GLVis:
```sh
~> cd Laghos/rom/dependencies
~/Laghos/rom/dependencies> git clone https://github.com/GLVis/glvis.git ./glvis
~/Laghos/rom/dependencies> cd glvis/
~/Laghos/rom/dependencies/glvis> make
~/Laghos/rom/dependencies/glvis> cd ..
```
The easiest way to visualize Laghos results is to have GLVis running in a
separate terminal. Then the `-vis` option in Laghos will stream results directly
to the GLVis socket.

Build the LaghosROM
```sh
~> cd Laghos/rom
~/Laghos/rom> make
~/Laghos/rom> make merge
```
This can be followed by `make test` and `make install` to check and install the
build respectively. See `make help` for additional options.

To see the building instruction of **Laghos**, please see README.md in Laghos directory.

## Running

#### Gresho vortex problem

The 2D Gresho vortex problem can be runned with `-p 4`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -tf 0.62 -s 7 -offline -writesol -romsvds -rostype load -romsns -nwinsamp 21 -sample-stages 
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -tf 0.62 -s 7 -online -romhrprep -rostype load -romsns -nwin 152 -sfacv 8 -sface 8
./laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -tf 0.62 -s 7 -online -romhr -rostype load -romsns -nwin 152 -sfacv 8 -sface 8
./laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -tf 0.62 -s 7 -soldiff -restore -rostype load -romsns -nwin 152
```

#### Sedov blast problem

The 3D Sedov blast wave problem can be runned with `-p 1`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -offline -romsvds -writesol -rostype load -romsns -nwinsamp 21 -sample-stages
```
The corresponding run of the reduce order model is:
```sh
./laghos -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -online -rostype load -romhrprep -romsns -nwin 66 -sfacv 2 -sface 2
./laghos -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -online -rostype load -romhr -romsns -nwin 66 -sfacv 2 -sface 2
./laghos -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -restore -soldiff -rostype load -romsns -nwin 66
```

#### Taylor-Green vortex problem

The 3D Taylor-Green vortex problem can be runned with `-p 0`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 0 -rs 2 -cfl 0.1 -tf 0.25 -s 7 -offline -romsvds -writesol -rostype load -romsns -nwinsamp 21 -sample-stages
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 0 -rs 2 -cfl 0.1 -tf 0.25 -s 7 -online -rostype load -romhrprep -romsns -nwin 82 -sfacv 2 -sface 2
./laghos -p 0 -rs 2 -cfl 0.1 -tf 0.25 -s 7 -online -rostype load -romhr -romsns -nwin 82 -sfacv 2 -sface 2
./laghos -p 0 -rs 2 -cfl 0.1 -tf 0.25 -s 7 -restore -soldiff -rostype load -romsns -nwin 82
```

#### Triple-point problem

The 3D triple-point problem can be runned with `-p 3`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -s 7 -cfl 0.5 -offline -writesol -romsvds -rostype load -romsns -nwinsamp 21 -sample-stages
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -s 7 -cfl 0.5 -online -romhrprep -rostype load -romsns -nwin 18 -sfacv 2 -sface 2
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -s 7 -cfl 0.5 -online -romhr -rostype load -romsns -nwin 18 -sfacv 2 -sface 2
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -s 7 -cfl 0.5 -restore -soldiff -rostype load -romsns -nwin 18
```

#### Parametric Sedov blast problem

The 3D parametric Sedov blast wave problem can be runned with `-p 1` in the snapshot collection. The snapshot data is passed to `merge`. (Note the `-bef` and `-rpar` option.)

A sample run of the offline stage is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -offline -romsns -rostype interpolate -sample-stages -sdim 10000 -bef 1.0 -rpar 0
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -offline -romsns -rostype interpolate -sample-stages -sdim 10000 -bef 1.2 -rpar 1
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -offline -romsns -rostype interpolate -sample-stages -sdim 10000 -bef 0.8 -rpar 2
./merge -nset 3 -romsns -rostype interpolate -nwinsamp 20 
```

A sample run of the full order model is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -writesol -bef 0.9 
```

A corresponding sample run of the reduce order model is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -online -romhrprep -romsns -rostype interpolate -nwin 80 -sfacv 4 -sface 4 -bef 0.9
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -online -romhr -romsns -rostype interpolate -nwin 80 -sfacv 4 -sface 4 -bef 0.9
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.8 -s 7 -restore -soldiff -romsns -rostype interpolate -nwin 80
```

## Verification of Results

To make sure the results are correct, we tabulate reference final 
L2 relative error for velocity (`E_v`), energy (`E_e`), position (`E_x`) 
between the full order model and the reduced order model (note the `-writesol` and `-soldiff` option):

| `Problem` | `E_v` | `E_e` | `E_x` |
| ----- | ------ | ---- | --- |
| Gresho vortex | 4.44385e-05 | 3.9624e-06 | 1.3527e-06 |
| Sedov blast | 2.17542e-04 | 2.35066e-04 | 2.25713e-05 |
| Taylor-Green vortex | 1.13658e-06 | 1.00322e-07 | 1.84963e-08 |
| Triple-point | 8.09275e-04 | 2.77958e-04 | 3.14557e-05 |
| Parametric Sedov blast | 4.32075e-02 | 9.04933e-03 | 4.34223e-03 |

The results are generated by the [Quartz](https://hpc.llnl.gov/hardware/platforms/Quartz) machine at LLNL.

## Contact

You can reach the Laghos ROM team by emailing surrogate@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Laghos/issues).

## libROM: library for Reduced Order Model

We heavily rely on the library for reduced order model package,
[libROM](libROM.net), developped by ROM team at [LLNL](https://www.llnl.gov).

## YouTube Tutorials on Reduced Order Models

These YouTube tutorial videos might be useful for you:

- [Poisson equation and its finite element discretization](https://youtu.be/YaZPtlbGay4)
- [Poisson equation and its reduced order model](https://youtu.be/YlFrBP31riA)

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.
