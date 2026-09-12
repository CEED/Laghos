#!/bin/bash
# Multi-window triple point run for BEQP and CEQP.
# Old-job settings (mesh, refinement, cfl) with a shortened time horizon.

set -e

P="-p 3 -m data/box01_hex.mesh -rs 2 -cfl 0.5 -s 7 -tf 0.2"
RUN="srun"

# $1 = sampling type (eqp | eqp_energy), $2 = output dir,
# $3 = ranks for stages 1, 2, 3, 5 (stage 4 is always 1).
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
  $RUN -n $NP ./merge -o $OUT -nset 1 -romsns $OS_OFF -eqp \
       -nwinsamp 10 -ef 0.9999

  # Number of windows produced by the merge (one twp line per window).
  NWIN=$(wc -l < run/$OUT/twp.csv | tr -d ' ')
  echo "Windows: $NWIN"
  # Guard: -nwin 0 would fail confusingly in every later stage.
  if [ "$NWIN" -lt 1 ]; then
    echo "ERROR: run/$OUT/twp.csv is empty; merge produced no windows" >&2
    exit 1
  fi

  # 3. Online prep (parallel)
  $RUN -n $NP laghos -o $OUT $P -online -romhrprep -romsns \
       $OS_ON -no-romgs -nwin $NWIN -hrsamptype $TYPE -lqnnls -maxnnls 500

  # 4. Online: always -n 1.
  # The sample mesh lives on rank 0 and -n>1 deadlocks in readSP.
  $RUN -n 1 laghos -o $OUT $P -online -romhr -romsns \
       $OS_ON -no-romgs -nwin $NWIN -hrsamptype $TYPE -lqnnls

  # 5. Restore
  $RUN -n $NP laghos -o $OUT $P -restore -soldiff -romsns \
       $OS_ON -nwin $NWIN -hrsamptype $TYPE
}

# Single rank runs
run_pipeline eqp        triple_eqp_mw_1r   1
run_pipeline eqp_energy triple_ceqp_mw_1r  1

# Multi rank runs.
# Rank shape is N/N/N/1/N: stages 1, 2, 3, 5 share one N, stage 4 is
# always serial.
# Keep the separate _2r output dirs; the bases are one file per rank.
run_pipeline eqp        triple_eqp_mw_2r   2
run_pipeline eqp_energy triple_ceqp_mw_2r  2
