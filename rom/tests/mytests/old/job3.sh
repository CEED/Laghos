#!/bin/bash
#SBATCH --mail-user=cv1038@unh.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --job-name="CEQP-R3"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00

srun laghos -p 3 -m data/box01_hex.mesh -tf 0.8 -visit -vs 1000 --visitfilename visit -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp_energy -nwinsamp 10 -sample-stages > ceqp_3a.out

srun laghos -p 3 -m data/box01_hex.mesh -tf 0.8 -visit -vs 1000 --visitfilename visit -s 7 -online -romhr --no-romoffset --no-romsns -hrsamptype eqp_energy -nwin 39 > ceqp_3b.out
