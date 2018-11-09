               __                __
              / /   ____  ____  / /_  ____  _____
             / /   / __ `/ __ `/ __ \/ __ \/ ___/
            / /___/ /_/ / /_/ / / / / /_/ (__  )
           /_____/\__,_/\__, /_/ /_/\____/____/
                       /____/

        High-order Lagrangian Hydrodynamics Miniapp

                       AMR version


## Overview

This directory contains the automatic mesh refinement (AMR) version of **Laghos**
(LAGrangian High-Order Solver), which is currently considered experimental and
is NOT the official benchmark version of the miniapp.

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


## Differences from the official benchmark version

The AMR version differs from the official benchmark version of Laghos (in the
top-level directory) in the following ways:

1. The `-amr` parameter turns on dynamic AMR.
2. The code includes functionality to change the mesh and the hydro operator on
   the fly.
3. Parallel partitioning and load balancing is based on MFEM's non-conforming
   mesh algorithm that partitions a space-filling curve. METIS is not required.


## Limitations

- The current AMR implementation is just a demonstration.
- Only the Sedov problem is supported, as the refinement/derefinement decisions
  are very simple and tailored specifically to Sedov.
- Partial assembly is currently not supported in AMR mode. Also, the hydro
  operator update is currently not efficient (e.g., the whole mass matrix is
  reassembled on each mesh change).
- MFEM currently does not support derefinement interpolation for non-nodal bases.
  The AMR version therefore does not use `BasisType::Positive` for the L2 space.


## Building

The AMR version can be built following the same [instructions](../README.md) as
for the top-level directory.


## Running

The AMR version only runs with problem 1 (Sedov blast). New parameters are:

- `-amr`: turn on AMR mode
- `-rt` or `--ref-threshold`: tweak the refinement threshold
- `-dt` or `--deref-threshold`: tweak the derefinement threshold

One of the sample runs is:
```sh
mpirun -np 8 laghos -p 1 -m ../data/cube01_hex.mesh -rs 4 -tf 0.6 -rt 1e-3 -amr
```

This produces the following plots at steps 900 and 2463:

<table border="0">
<td><img src="data/sedov-amr-900.png">
<td><img src="data/sedov-amr-2463.png">
</table>


## Verification of Results

To make sure the results are correct, we tabulate reference final iterations
(`step`), time steps (`dt`) and energies (`|e|`) for the runs listed below:

1. `mpirun -np 8 laghos -p 1 -m ../data/square01_quad.mesh -rs 4 -tf 0.8 -amr`
2. `mpirun -np 8 laghos -p 1 -m ../data/square01_quad.mesh -rs 4 -tf 0.8 -ok 3 -ot 2 -amr`
3. `mpirun -np 8 laghos -p 1 -m ../data/cube01_hex.mesh -rs 3 -tf 0.6 -amr`
4. `mpirun -np 8 laghos -p 1 -m ../data/cube01_hex.mesh -rs 4 -tf 0.6 -rt 1e-3 -amr`

| run | `step` | `dt` | `e` |
| --- | ------ | ---- | ----- |
|  1. | 2374 | 0.000308 | 90.9397751791 |
|  2. | 2727 | 0.000458 | 168.0063715464 |
|  3. |  998 | 0.001262 | 388.6322346715 |
|  4. | 2463 | 0.000113 | 1703.2772575684 |

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
