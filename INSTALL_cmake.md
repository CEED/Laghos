### CMake

This installs built dependencies to `INSTALLDIR` using the `CC` C-compiler and `CXX` C++17-compiler.

Build METIS (optional):
```sh
git clone https://github.com/KarypisLab/METIS.git
cd METIS
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$CC -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
make -j install
```
For large runs (problem size above 2 billion unknowns), add `-DMETIS_USE_LONGINDEX=ON` option to the above `cmake` line. If building without METIS only Cartesian partitioning is supported.

Build Umpire (CUDA, optional):
```sh
git clone https://github.com/LLNL/Umpire.git
cd Umpire
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DENABLE_CUDA=ON -DUMPIRE_ENABLE_C=ON -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CUDACC
make -j install
```

Build Umpire (HIP, optional):
```sh
git clone https://github.com/LLNL/Umpire.git
cd Umpire
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=native -DENABLE_HIP=ON -DUMPIRE_ENABLE_C=ON -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_HIP_COMPILER=$HIPCC
make -j install
```

Build *hypre* (CPU-only):

```sh
git clone https://github.com/hypre-space/hypre.git
cd hypre/build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
make -j install
```

Build *hypre* (CUDA):

```sh
git clone https://github.com/hypre-space/hypre.git
cd hypre/build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CUDACC -DHYPRE_ENABLE_GPU_AWARE_MPI=ON -DHYPRE_ENABLE_UMPIRE=ON
make -j install
```
``HYPRE_ENABLE_GPU_AWARE_MPI`` and ``HYPRE_ENABLE_UMPIRE`` may be optionally turned off.

Build *hypre* (HIP):

```sh
git clone https://github.com/hypre-space/hypre.git
cd hypre/build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DHYPRE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=native -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_HIP_COMPILER=$HIPCC -DHYPRE_ENABLE_GPU_AWARE_MPI=ON -DHYPRE_ENABLE_UMPIRE=ON
make -j install
```
``HYPRE_ENABLE_GPU_AWARE_MPI`` and ``HYPRE_ENABLE_UMPIRE`` may be optionally turned off.
For large runs (problem size above 2 billion unknowns), enable the `HYPRE_ENABLE_MIXEDINT` option.

Build MFEM (CPU-only):
```sh
git clone https://github.com/mfem/mfem.git
cd mfem
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DHYPRE_DIR=$INSTALLDIR -DMETIS_DIR=$INSTALLDIR -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DCMAKE_CXX_COMPILER=$CXX
make -j install
```
`MFEM_USE_METIS` may be optionally disabled.
See the [MFEM building page](http://mfem.org/building/) for additional details.

Build MFEM (CUDA):
```sh
git clone https://github.com/mfem/mfem.git
cd mfem
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DHYPRE_DIR=$INSTALLDIR -DMETIS_DIR=$INSTALLDIR -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DMFEM_USE_CUDA=ON -DMFEM_USE_UMPIRE=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CUDACC -DUMPIRE_DIR=$INSTALLDIR
make -j install
```
`MFEM_USE_METIS` and `MFEM_USE_UMPIRE may be optionally disabled.
See the [MFEM building page](http://mfem.org/building/) for additional details.

Build MFEM (HIP):
```sh
git clone https://github.com/mfem/mfem.git
cd mfem
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DHYPRE_DIR=$INSTALLDIR -DMETIS_DIR=$INSTALLDIR -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DMFEM_USE_HIP=ON -DMFEM_USE_UMPIRE=ON -DCMAKE_HIP_ARCHITECTURES=native -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_HIP_COMPILER=$HIPCC -DUMPIRE_DIR=$INSTALLDIR
make -j install
```
`MFEM_USE_METIS` and `MFEM_USE_UMPIRE` may be optionally disabled.
See the [MFEM building page](http://mfem.org/building/) for additional details.

GLVis (optional):
```sh
git clone https://github.com/GLVis/glvis.git
cd glvis
mkdir build
cd build
cmake .. -DMFEM_DIR=$INSTALLDIR -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_CXX_COMPILER=$CXX
```
The easiest way to visualize Laghos results is to have GLVis running in a
separate terminal. Then the `-vis` option in Laghos will stream results directly
to the GLVis socket.

Caliper (Optional):
1. Clone and build Adiak:
```sh
git clone --recursive https://github.com/LLNL/Adiak.git
cd Adiak
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=On -DENABLE_MPI=On \
      -DCMAKE_INSTALL_PREFIX=$INSTALLDIR ..
make -j install
```
2. Clone and build Caliper:
```sh
git clone https://github.com/LLNL/Caliper.git
cd Caliper
mkdir build && cd build
cmake -DWITH_MPI=True -DWITH_ADIAK=True -Dadiak_ROOT=$INSTALLDIR \
      -DCMAKE_INSTALL_PREFIX=$INSTALLDIR ..
make -j install
```

Laghos (CPU-only):
```sh
git clone https://github.com/CEED/Laghos.git
cd Laghos
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_CXX_COMPILER=$CXX -Dcaliper_ROOT=$INSTALLDIR -DLAGHOS_USE_CALIPER=ON
make -j
```
`LAGHOS_USE_CALIPER` may be optionally disabled.

Laghos (CUDA):
```sh
git clone https://github.com/CEED/Laghos.git
cd Laghos
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$CUDACC -DCMAKE_CUDA_ARCHITECTURES=native -Dcaliper_ROOT=$INSTALLDIR -DLAGHOS_USE_CALIPER=ON
make -j
```
`LAGHOS_USE_CALIPER` may be optionally disabled.

Laghos (HIP):
```sh
git clone https://github.com/CEED/Laghos.git
cd Laghos
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_HIP_COMPILER=$HIPCC -DCMAKE_HIP_ARCHITECTURES=native -Dcaliper_ROOT=$INSTALLDIR -DLAGHOS_USE_CALIPER=ON
make -j
```
`LAGHOS_USE_CALIPER` may be optionally disabled.
