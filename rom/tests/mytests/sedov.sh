#!/bin/bash
# Multi-window Sedov run for BEQP and CEQP.

set -e

P="-m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.08 -s 7"
RUN="srun"

# $1 = sampling type (eqp | eqp_energy), $2 = output dir, $3 = prep ranks
run_pipeline () {
  TYPE=$1; OUT=$2; NP=$3

  if [ "$TYPE" = "eqp_energy" ]; then
    OS_OFF="-romos -rostype load"        # offline and merge
    OS_ON="-no-romoffset -rostype load"  # online prep / online hr / restore
  else
    OS_OFF="-romos -rostype load"
    OS_ON="-romos -rostype load"
  fi

  # 1. Offline
  $RUN -n $NP laghos -o $OUT $P -offline -romsns $OS_OFF \
       -rpar 0 -sample-stages -sdim 1000 -writesol

  # 2. Merge
  $RUN -n $NP ./merge -o $OUT -nset 1 -romsns $OS_OFF -eqp -nwinsamp 10

  # 3. Online prep (parallel)
  $RUN -n $NP laghos -o $OUT $P -online -romhrprep -romsns \
       $OS_ON -no-romgs -nwin 4 -hrsamptype $TYPE -lqnnls -maxnnls 500

  # 4. Online (serial: sample mesh is on rank 0).
  $RUN -n 1 laghos -o $OUT $P -online -romhr -romsns \
       $OS_ON -no-romgs -nwin 4 -hrsamptype $TYPE -lqnnls

  # 5. Restore
  $RUN -n $NP laghos -o $OUT $P -restore -soldiff -romsns \
       $OS_ON -nwin 4 -hrsamptype $TYPE
}

# Signle rank runs
run_pipeline eqp        sedov_eqp_mw_1r   1
run_pipeline eqp_energy sedov_ceqp_mw_1r  1

# Multi rank runs
#run_pipeline eqp        sedov_eqp_mw_2r   2
#run_pipeline eqp_energy sedov_ceqp_mw_2r  2
