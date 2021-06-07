The greedy algorithm is obtained from algorithm 2 of Choi et. al's
paper "Gradient-based constrained optimization using a database
of linear reduced-order models": https://arxiv.org/abs/1506.07849.
To build an accurate ROM database for a finite simulation parameter space
with Laghos, use one of the following commands.

Hyperreduction is currently not supported (until next PR).

To run locally without using srun:

./scripts/laghos_greedy_alg_launcher.sh laghos ... (to run locally without using srun)

To use sbatch, maintaining one node throughout the algorithm. sbatch configuration
commands can be modified within the launcher script:

sbatch scripts/laghos_greedy_alg_launcher.sh srun -n 1 -p pdebug laghos ...

Example command with all available options:

./scripts/laghos_greedy_alg_launcher.sh laghos -m data/cube01_hex.mesh -pt 211 -tf 0.01 -build-database -writesol -romsvds -greedy-param "bef" -greedy-param-min 0.5 -greedy-param-max 2.5 -greedy-param-size 5 -greedysubsize 2 -greedyconvsize 3 -greedytol 0.1 -greedyalpha 1.05 - greedymaxclamp 2.0 -greedysamptype random -greedyerrindtype useLastLifted

- greedy-param (default: "bef"): The value to change (i.e. blast-energyfactor, etc.)
New parameters can easily be added by adding an entry to the map in getGreedyParam()
in laghos_rom.hpp. Ideally, these parameters should exist in ROM_Options.
- greedy-param-min (default: 0): The minimum value of the parameter domain.
- greedy-param-max (default: 0): The maximum value of the parameter domain.
- greedy-param-size (default: 0): The maximum number of points to sample.
- greedysubsize (default: 0): The number of points to obtain an error indicator at
each iteration. Increase to obtain better guarantee that relative error tolerance
is met across all points in the parameter space.
- greedyconvsize (default: 0): The number of random points to check the error indicator to
figure out convergence has been reached (i.e. relative error tolerance is met
at all points within the domain.) Increase to obtain better guarantee that relative
error tolerance is met across all points in the parameter space.
- greedytol (default: 0.1): The relative error tolerance.
- greedyalpha (default: 1.05): The minimum factor to increase the error indicator
tolerance by each iteration. This makes more sense once you run the algorithm with
default parameters and see the output.
- greedymaxclamp (default: 2): The maximum factor to increase the error indicator
tolerance by each iteration. This makes more sense once you run the algorithm with
default parameters and see the output.
- greedysamptype (default: "random"): The sampling type. Choose from "random" or "latin-hypercube".
- greedyerrindtype (default: "useLastLifted"): The error indicator type. Choose from "useLastLifted" and "varyBasisSize".
