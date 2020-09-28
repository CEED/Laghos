               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp

                     SERIAL version

## Overview

This directory contains the SERIAL version of the **Laghos** (LAGrangian
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

The serial version differs from the official benchmark version of Laghos (in the
top-level directory) in the following ways:

1. No differences.

## Building

The serial version of Laghos has the following external dependencies:

- MFEM, used for (high-order) finite element discretization.

To build the miniapp, first clone and build MFEM:
```sh
~> git clone https://github.com/mfem/mfem.git ./mfem
~> cd mfem/
~/mfem> make serial -j
~/mfem> cd ..
```
See the [MFEM building page](http://mfem.org/building/) for additional details.

(Optional) Clone and build GLVis:
```sh
~> git clone https://github.com/GLVis/glvis.git ./glvis
~> cd glvis/
~/glvis> make
~/glvis> cd ..
```
The easiest way to visualize Laghos results is to have GLVis running in a
separate terminal. Then the `-vis` option in Laghos will stream results directly
to the GLVis socket.

Build Laghos
```sh
~> cd Laghos/serial
~/serial> make
```
This can be followed by `make test` and `make install` to check and install the
build respectively. See `make help` for additional options.

## Running

The serial version can run the same examples as the official benchmark version
of Laghos, without MPI parallelization.

## Verification of Results


To make sure the results are correct, we tabulate reference final iterations
(`step`), time steps (`dt`) and energies (`|e|`) for the runs listed below:

1. `./laghos -p 0 -dim 2 -rs 3 -tf 0.75 -pa`
2. `./laghos -p 0 -dim 3 -rs 1 -tf 0.75 -pa`
3. `./laghos -p 1 -dim 2 -rs 3 -tf 0.8 -pa`
4. `./laghos -p 1 -dim 3 -rs 2 -tf 0.6 -pa`
5. `./laghos -p 2 -dim 1 -rs 5 -tf 0.2 -fa`
6. `./laghos -p 3 -m ../data/rectangle01_quad.mesh -rs 2 -tf 3.0 -pa`
7. `./laghos -p 3 -m ../data/box01_hex.mesh -rs 1 -tf 3.0 -pa`
8. `./laghos -p 4 -m ../data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62831853 -s 7 -pa`

| `run` | `step` | `dt` | `e` |
| ----- | ------ | ---- | --- |
|  1. |  339 | 0.000702 | 4.9695537349e+01 |
|  2. | 1041 | 0.000121 | 3.3909635545e+03 |
|  3. | 1154 | 0.001655 | 4.6303396053e+01 |
|  4. |  560 | 0.002449 | 1.3408616722e+02 |
|  5. |  413 | 0.000470 | 3.2012077410e+01 |
|  6. | 2872 | 0.000064 | 5.6547039096e+01 |
|  7. |  528 | 0.000180 | 5.6505348812e+01 |
|  8. |  776 | 0.000045 | 4.0982431726e+02 |

Similar GPU runs using the MFEM CUDA device can be run as follows:

1. `./laghos -p 0 -dim 2 -rs 3 -tf 0.75 -pa -d cuda`
2. `./laghos -p 0 -dim 3 -rs 1 -tf 0.75 -pa -d cuda`
3. `./laghos -p 1 -dim 2 -rs 3 -tf 0.80 -pa -d cuda`
4. `./laghos -p 1 -dim 3 -rs 2 -tf 0.60 -pa -d cuda`
5. `./laghos -p 2 -dim 1 -rs 5 -tf 0.20 -fa`
6. `./laghos -p 3 -m ../data/rectangle01_quad.mesh -rs 2 -tf 3.0 -pa -d cuda`
7. `./laghos -p 3 -m ../data/box01_hex.mesh -rs 1 -tf 3.0 -pa -cgt 1e-12 -d cuda`
8. `./laghos -p 4 -m ../data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62831853 -s 7 -pa -d cuda`

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
