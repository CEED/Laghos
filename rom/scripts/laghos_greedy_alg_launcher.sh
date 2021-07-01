#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p pdebug
#SBATCH -o greedy.log
#SBATCH --open-mode truncate

date
PARALLEL_PROCESSORS=8
# Save directory of this script
if [[ -f "$PWD/scripts/laghos_greedy_alg_launcher.sh" ]]; then
        SCRIPT_DIR=$PWD
else
        echo "Run tests from the Laghos directory"
fi

# Check machine
case "$(uname -s)" in
    Linux*)
 PARALLEL_COMMAND="srun -p pdebug -n $PARALLEL_PROCESSORS $SCRIPT_DIR/$@"
          SERIAL_COMMAND="srun -p pdebug -n 1 $SCRIPT_DIR/$@";;
    Darwin*)
 PARALLEL_COMMAND="mpirun -oversubscribe -n $PARALLEL_PROCESSORS $SCRIPT_DIR/$@"
          SERIAL_COMMAND="mpirun -oversubscribe -n 1 $SCRIPT_DIR/$@";;
    *)
 echo "The regression tests can only run on Linux and MAC."
 exit 1
esac

# Check if greedy algorithm has ended
while [ "$?" -eq 0 ]
do
    if [[ -f "$SCRIPT_DIR/run/hyperreduce.txt" ]]; then
        echo "$SERIAL_COMMAND"
        eval "$SERIAL_COMMAND"
        rm $SCRIPT_DIR/run/hyperreduce.txt
    else
        echo "$PARALLEL_COMMAND"
        eval "$PARALLEL_COMMAND"
    fi
done
date
