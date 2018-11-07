               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp

                      OCCA version

## Overview

This directory contains the OCCA version of the **Laghos** (LAGrangian
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

The OCCA version differs from the official benchmark version of Laghos (in the
top-level directory) in the following ways:

1. Only problems 0 and 1 are defined
2. Final iterations (`step`), time steps (`dt`) and energies (`|e|`) differ from the original version

## Building

Follow the steps below to build the OCCA version

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
-   ``make config shared=1 prefix=`pwd` ``
-   `make && make install`
-   `cd ..`

### OCCA
-   `git clone git@github.com:libocca/occa.git`
-   `cd occa`
-   `make -j`
-   `cd ..`

### MFEM with OCCA
-   `git clone git@github.com:mfem/mfem.git`
-   `cd mfem`
-   `git checkout occa-dev`
-   ``make config MFEM_USE_MPI=YES HYPRE_DIR=`pwd`/../hypre-2.11.2/src/hypre MFEM_USE_METIS_5=YES METIS_DIR=`pwd`/../metis-5.1.0 MFEM_USE_OCCA=YES OCCA_DIR=`pwd`/../occa``
-   `make status` to verify that all the include paths are correct
-   `make -j`
-   `cd ..`

### OCCA Laghos
-   `git clone git@github.com:CEED/Laghos.git`
-   `cd Laghos/occa`
-   `make -j`

## Running

The OCCA version can run the same sample test runs as the official benchmark
version of Laghos.

### Options
-   -m <string>: Mesh file to use
-   -ok <int>: Order (degree) of the kinematic finite element space
-   -rs <int>: Number of times to refine the mesh uniformly in serial
-   -p <int>: Problem setup to use, Sedov problem is '1'
-   -cfl <double>: CFL-condition number
-   -ms <int>: Maximum number of steps (negative means no restriction)
-   -d <string>: OCCA device string (e.g. "mode: 'CUDA', device_id: 0")

## Verification of Results

TBD

## Contact

You can reach the Laghos team by emailing laghos@llnl.gov or by leaving a
comment in the [issue tracker](https://github.com/CEED/Laghos/issues).

## Copyright

The following copyright applies to each file in the CEED software suite,
unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE in the top-level directory for details.
