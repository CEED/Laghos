#!/bin/bash
#SBATCH --mail-user=cv1038@unh.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --job-name="CEQP-R4"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00

srun laghos -p 0 -m data/cube01_hex.mesh -cfl 0.15 -tf 0.25 -visit -vs 1000 --visitfilename visit -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp_energy -nwinsamp 10 -sample-stages > ceqp_4a.out

srun laghos -p 0 -m data/cube01_hex.mesh -cfl 0.15 -tf 0.25 -visit -vs 1000 --visitfilename visit -s 7 -online -romhr --no-romoffset --no-romsns -hrsamptype eqp_energy -nwin 120 > ceqp_4b.out
