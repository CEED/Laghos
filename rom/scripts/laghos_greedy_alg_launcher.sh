#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p pdebug
#SBATCH -o greedy.log
#SBATCH --open-mode truncate

PARALLEL=${PARALLEL:-1}

# Check machine
case "$(uname -s)" in
    Linux*)
		  PARALLEL_COMMAND="srun -p pdebug -n $PARALLEL $@"
          SERIAL_COMMAND="srun -p pdebug -n 1 $@";;
    Darwin*)
		  PARALLEL_COMMAND="mpirun -oversubscribe -n $PARALLEL $@"
          SERIAL_COMMAND="mpirun -oversubscribe -n 1 $@";;
    *)
		  echo "The regression tests can only run on Linux and MAC."
		  exit 1
esac

# Check if greedy algorithm has ended
while [ "$?" -eq 0 ]
do
    if [[ -f "../run/hyperreduce.txt" ]]; then
        "$SERIAL_COMMAND"
        rm ../run/hyperreduce.txt
    else
        "$PARALLEL_COMMAND"
    fi
done
exit 0
