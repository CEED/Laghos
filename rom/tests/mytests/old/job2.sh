#!/bin/bash
#SBATCH --mail-user=cv1038@unh.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --job-name="CEQP-S2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00

srun laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 2 -ot 1 -cfl 0.35 -tf 0.4 -visit -vs 1000 --visitfilename visit -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load -no-romsns -hrsamptype eqp_energy -nwinsamp 10 -sample-stages > ceqp_2a.out

srun laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 2 -ot 1 -cfl 0.35 -tf 0.4 -visit -vs 1000 --visitfilename visit -s 7 -online --no-romoffset -romhr --no-romsns -hrsamptype eqp_energy -nwin 132 > ceqp_2b.out
