#!/bin/bash
#--------------------------------------------------------------------------------
# mpirun -np $nmpi bind.sh your.exe [args]
# optionally set BIND_BASE in your job script
# optionally set BIND_STRIDE in your job script
# optionally set BIND_POLICY=packed in your job script
# optionally set BIND_CPU_LIST="cpu0 cpu1 ..." in your job script
# Note : for some OpenMP implementations (GNU OpenMP) use mpirun --bind-to none
#--------------------------------------------------------------------------------
cpus_per_node=`cat /proc/cpuinfo | grep processor | wc -l`

if [ -z "$OMPI_COMM_WORLD_LOCAL_SIZE" ]; then

  let OMPI_COMM_WORLD_LOCAL_SIZE=1

  let OMPI_COMM_WORLD_LOCAL_RANK=0

fi

# if OMP_NUM_THREADS is not set, assume no threading and bind with taskset
if [ -z "$OMP_NUM_THREADS" ]; then

  if [ "$OMPI_COMM_WORLD_RANK" == "0" ]; then
    echo bind.sh: OMP_NUM_THREADS is not set ... assuming one thread
  fi

  if [ -z "$BIND_CPU_LIST" ]; then

    if [ -z "$BIND_BASE" ]; then
      let BIND_BASE=0
    fi

    if [ -z "$BIND_STRIDE" ]; then
      let cpus_per_rank=$cpus_per_node/$OMPI_COMM_WORLD_LOCAL_SIZE
    else
      let cpus_per_rank=$BIND_STRIDE
    fi

    let start_cpu=$BIND_BASE+$OMPI_COMM_WORLD_LOCAL_RANK*$cpus_per_rank

    let stop_cpu=$start_cpu+$cpus_per_rank-1

    if [ "${BIND_ALL}" == "yes" ]; then
      printf -v command "exec taskset -c %d-%d"  $start_cpu  $stop_cpu
    else
      printf -v command "exec taskset -c %d"  $start_cpu
    fi

    $command "$@"

  else

    declare -a cpulist=($BIND_CPU_LIST)

    exec taskset -c ${cpulist[$OMPI_COMM_WORLD_LOCAL_RANK]} "$@"

  fi

else
#if OMP_NUM_THREADS is set, bind using OMP_PLACES

  if [ -z "$BIND_STRIDE" ]; then
    let cpus_required=$OMP_NUM_THREADS*$OMPI_COMM_WORLD_LOCAL_SIZE
    let BIND_STRIDE=$cpus_per_node/$cpus_required
  fi

  if [ "${BIND_POLICY}" == "packed" ]; then
    let cpus_per_rank=$OMP_NUM_THREADS*$BIND_STRIDE
  else
    let cpus_per_rank=$cpus_per_node/$OMPI_COMM_WORLD_LOCAL_SIZE
  fi

  if [ -z "$BIND_BASE" ]; then
    let base=0;
  else
    let base=$BIND_BASE;
  fi

  let start_cpu=$base+$OMPI_COMM_WORLD_LOCAL_RANK*$cpus_per_rank

  let stop_cpu=$start_cpu+$OMP_NUM_THREADS*$BIND_STRIDE-1

  export OMP_PLACES={$start_cpu:$OMP_NUM_THREADS:$BIND_STRIDE}

  export OMP_PROC_BIND=true

  if [ "${BIND_ALL}" == "yes" ]; then
    printf -v command "exec taskset -c %d-%d"  $start_cpu  $stop_cpu
    $command "$@"
  else
    exec "$@"
  fi

fi
