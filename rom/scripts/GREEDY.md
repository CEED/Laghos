To use the greedy algorithm obtained from algorithm 2 of Choi et. al's
paper "Gradient-based constrained optimization using a database
of linear reduced-order models": https://arxiv.org/abs/1506.07849
to build a ROM database with Laghos, use one of the following commands.

To run locally without using srun:

scripts/laghos_greedy_alg_launcher.sh ./laghos ... (to run locally without using srun)

To use srun, however a new node is obtained each iteration:

scripts/laghos_greedy_alg_launcher.sh srun -n 1 -p pdebug laghos ...

To use sbatch, maintaining one node throughout the algorithm. sbatch configuration
commands can be modified within the launcher script:

sbatch scripts/laghos_greedy_alg_launcher.sh srun -n 1 -p pdebug laghos ...
