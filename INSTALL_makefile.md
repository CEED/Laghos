### Makefile

To build the miniapp, first download *hypre* and METIS from the links above
and put everything on the same level as the `Laghos` directory:
```sh
~> ls
Laghos/  v2.11.2.tar.gz  metis-4.0.3.tar.gz
```

Build *hypre*:
```sh
~> tar -zxvf v2.11.2.tar.gz
~> cd hypre-2.11.2/src/
~/hypre-2.11.2/src> ./configure --disable-fortran
~/hypre-2.11.2/src> make -j
~/hypre-2.11.2/src> cd ../..
~> ln -s hypre-2.11.2 hypre
```
For large runs (problem size above 2 billion unknowns), add the
`--enable-bigint` option to the above `configure` line.

Build METIS:
```sh
~> tar -zxvf metis-4.0.3.tar.gz
~> cd metis-4.0.3
~/metis-4.0.3> make
~/metis-4.0.3> cd ..
~> ln -s metis-4.0.3 metis-4.0
```
This build is optional, as MFEM can be build without METIS by specifying
`MFEM_USE_METIS = NO` below.

Clone and build the parallel version of MFEM:
```sh
~> git clone https://github.com/mfem/mfem.git ./mfem
~> cd mfem/
~/mfem> git checkout master
~/mfem> make parallel -j
~/mfem> cd ..
```
The above uses the `master` branch of MFEM.
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

(Optional) Build Caliper
1. Clone and build Adiak:
```sh
~> git clone --recursive https://github.com/LLNL/Adiak.git
~> cd Adiak
~/Adiak> mkdir build && cd build
~/Adiak> cmake -DBUILD_SHARED_LIBS=On -DENABLE_MPI=On \
               -DCMAKE_INSTALL_PREFIX=../../adiak ..
~/Adiak> make && make install
~/Adiak> cd ../..
```
2. Clone and build Caliper:
```sh
~> git clone https://github.com/LLNL/Caliper.git
~> cd Caliper
~/Caliper> mkdir build && cd build
~/Caliper> cmake -DWITH_MPI=True -DWITH_ADIAK=True -Dadiak_ROOT=../../adiak/ \
                 -DCMAKE_INSTALL_PREFIX=../../caliper ..
~/Caliper> make && make install
~/Caliper> cd ../..
```

Build Laghos
```sh
~> cd Laghos/
~/Laghos> make -j
```
This can be followed by `make test` and `make install` to check and install the
build respectively. See `make help` for additional options.

See also the `make setup` target that can be used to automated the
download and building of hypre, METIS and MFEM.
