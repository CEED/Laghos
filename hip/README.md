               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp

                      HIP version

## Overview

This directory contains the HIP version of the **Laghos** (LAGrangian
High-Order Solver), which is provided as a reference implementation and is NOT
the official benchmark version of the miniapp.

For more details about Laghos see the [README file](../README.md) in the
top-level directory.

The Laghos miniapp is part of the [CEED software suite](http://ceed.exascaleproject.org/software),
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

## Differences with the official benchmark version

The HIP version differs from the official benchmark version of Laghos (in the
top-level directory) in the following ways:

1. Only problems 0 and 1 are defined
2. Final iterations (`step`), time steps (`dt`) and energies (`|e|`) differ from the original version

## Building

Follow the steps below to build the HIP version with GPU acceleration.

### Environment setup
```sh
export MPI_HOME=~/usr/local/openmpi/3.0.0
```

### Hypre
- <https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods/download/hypre-2.11.2.tar.gz>
- `tar xzvf hypre-2.11.2.tar.gz`
- ` cd hypre-2.11.2/src`
- `./configure --disable-fortran --with-MPI --with-MPI-include=$MPI_HOME/include --with-MPI-lib-dirs=$MPI_HOME/lib`
- `make -j`
- `cd ../..`

### Metis
-   <http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz>
-   `tar xzvf metis-5.1.0.tar.gz`
-   `cd metis-5.1.0`
-   ``make config prefix=`pwd` ``
-   `make && make install`
-   `cd ..`

### MFEM
-   `git clone git@github.com:mfem/mfem.git`
-   `cd mfem`
-   `git checkout laghos-v2.0`
-   ``make config MFEM_USE_MPI=YES HYPRE_DIR=`pwd`/../hypre-2.11.2/src/hypre MFEM_USE_METIS_5=YES METIS_DIR=`pwd`/../metis-5.1.0``
-   `make status` to verify that all the include paths are correct
-   `make -j`
-   `cd ..`

### HIP Laghos
-   `git clone git@github.com:CEED/Laghos.git`
-   `cd Laghos/cuda`
-   edit the `makefile`, set HIP\_ARCH to the desired architecture and the absolute paths to HIP\_DIR, MFEM\_DIR, MPI\_HOME
-   `make` to build the HIP version

## Running

The HIP version can run the same sample test runs as the official benchmark
version of Laghos.

### Options
-   -m <string>: Mesh file to use
-   -ok <int>: Order (degree) of the kinematic finite element space
-   -rs <int>: Number of times to refine the mesh uniformly in serial
-   -p <int>: Problem setup to use, Sedov problem is '1'
-   -cfl <double>: CFL-condition number
-   -ms <int>: Maximum number of steps (negative means no restriction)
-   -aware: Enable or disable MPI HIP Aware

## Verification of Results

To make sure the results are correct, we tabulate reference final iterations
(`step`), time steps (`dt`) and energies (`|e|`) for the runs listed below:

1. `mpirun -np 4 laghos -p 0 -m ../data/square01_quad.mesh -rs 3 -tf 0.75 -pa`
2. `mpirun -np 4 laghos -p 0 -m ../data/cube01_hex.mesh -rs 1 -tf 0.75 -pa`
3. `mpirun -np 4 laghos -p 1 -m ../data/square01_quad.mesh -rs 3 -tf 0.8 -pa -cfl 0.05`
4. `mpirun -np 4 laghos -p 1 -m ../data/cube01_hex.mesh -rs 2 -tf 0.6 -pa -cfl 0.08`

| `run` | `step` | `dt` | `e` |
| ----- | ------ | ---- | --- |
|  1. |  333 | 0.000008 | 49.6955373330   |
|  2. | 1036 | 0.000093 | 3390.9635544029 |
|  3. | 1570 | 0.000768 | 46.2901037375   |
|  4. |  486 | 0.000864 | 135.1267396160  |

An implementation is considered valid if the final energy values are all within
round-off distance from the above reference values.

## Contact

You can reach the Laghos team by emailing laghos@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Laghos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE in the top-level directory for details.
