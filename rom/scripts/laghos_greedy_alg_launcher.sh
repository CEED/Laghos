#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p pdebug
#SBATCH -o greedy.log
#SBATCH --open-mode truncate

# Check if greedy algorithm has ended
while [ "$?" -eq 0 ]
do
    "$@"
done
exit 0
