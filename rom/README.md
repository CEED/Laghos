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

> D. Copeland, K. Huynh, S.W. Cheung, and Y. Choi <br>
> [Reduced order models for Lagrangian hydrodynamics]() <br>
> *In Preparation*, 2021.

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

#### Sedov blast

The 3D Sedov blast wave problem can be runned with `-p 1`.

A sample run of the offline stage is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.1 -pa -offline -romsvds -romos -rostype interpolate -romsrhs -bef 1.0 -rpar 0
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.1 -pa -offline -romsvds -romos -rostype interpolate -romsrhs -bef 1.2 -rpar 1
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.1 -pa -offline -romsvds -romos -rostype interpolate -romsrhs -bef 0.8 -rpar 2
./merge -nset 3 -ef 0.9999 -rhs -romos -rostype interpolate -nwinsamp 10
```

A sample run of the full order model is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.1 -pa -bef 1.1 -writesol
```
A corresponding sample run of the reduce order model is:
```sh
./laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.1 -online -romhr -romos -rostype interpolate -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.1 -nwin 27 -twp twpTemp.csv
```

#### Gresho vortex

The 2D Gresho vortex problem can be runned with `-p 4`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -pa -offline -romsvds -ef 0.9999 -nwinsamp 10 -romos -rostype load -romsrhs -writesol 
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -online -romhr -sfacv 32 -sface 32 -romsvds -nwin 76 -twp twpTemp.csv -romos -rostype load -romsrhs -romgs -soldiff
```

#### Taylor-Green vortex

The 3D Taylor-Green vortex problem can be runned with `-p 0`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 0 -m data/cube01_hex.mesh -rs 2 -cfl 0.1 -tf 0.25 -pa -offline -romsvds -ef 0.9999 -writesol -romos -rostype load -romsrhs -nwinsamp 10 
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 0 -m data/cube01_hex.mesh -rs 2 -cfl 0.1 -tf 0.25 -pa -online -soldiff -romsvds -romos -rostype load -romhr -romsrhs -romgs -sfacv 32 -sface 32 -twp twpTemp.csv -nwin 27 
```

#### Triple-point problem

The 3D triple-point problem can be runned with `-p 3`.

A sample run of the offline stage and the full order model is:
```sh
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -cfl 0.5 -pa -offline -writesol -romsvds -romos -rostype load -romsrhs -ef 0.9999 -nwinsamp 10 
```
The corresponding run of the reduce order model is:
```sh
./laghos -p 3 -m data/box01_hex.mesh -rs 2 -tf 0.8 -cfl 0.5 -pa -online -soldiff -nwin 20 -romhr -twp twpTemp.csv -romsvds -romos -rostype load -sfacv 1024 -sface 1024 -romgs -romsrhs
```

## Verification of Results

To make sure the results are correct, we tabulate reference final 
L2 relative error for velocity (`E_v`), energy (`E_e`), position (`E_x`) 
between the full order model and the reduced oder model (note the `-soldiff` option):

| `Problem` | `E_v` | `E_e` | `E_x` |
| ----- | ------ | ---- | --- |
| Sedov blast problem | 0.0739833401 | 0.0123365951 | 0.0008987601 |
| Gresho vortex | 0.0208559023 | 0.0003741972 | 0.0010241196 |
| Taylor-Green vortex | 0.0767346971 | 0.0023056170 | 0.0146010239 |
| Triple-point problem | 0.0026771894 | 0.0004677332 | 0.0000684808 |

The results are generated by an MacBook Air (13-inch, 2017) 
equipped with 1.8 GHz Dual-Core Intel Core i5. 

## Contact

You can reach the Laghos team by emailing laghos@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Laghos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.
