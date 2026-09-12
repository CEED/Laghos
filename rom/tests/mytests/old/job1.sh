#!/bin/bash
#SBATCH --mail-user=cv1038@unh.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --job-name="CEQP-S1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00

srun laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.3 -visit -vs 1000 --visitfilename visit -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp_energy -nwinsamp 10 -sample-stages > ceqp_1a.out

srun laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.3 -visit -vs 1000 --visitfilename visit -s 7 -online --no-romoffset -romhr --no-romsns -hrsamptype eqp_energy -nwin 91 > ceqp_1b.out
