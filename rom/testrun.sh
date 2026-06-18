#!/bin/bash
# Single-window Sedov validation for the energy-conserving EQP (CEQP)
# work, compared against the existing basic EQP method.
#
# Run from the ROM build directory (where laghos, merge and data/ live).
#
# Notes:
# - Stages 1, 2, 3, 5 run in parallel; stage 4 (online hyperreduced) runs
#   serially because the EQP sample mesh lives on rank 0.
# - The distributed mass Gram-Schmidt runs in stage 3 (romhrprep), so the
#   2-rank run exercises it. Comparing the 1-rank and 2-rank CEQP runs is
#   the rank-invariance check.
# - CEQP requires no SNS (-no-romsns) and zero offsets (-no-romoffset);
#   we also disable the runtime Gram-Schmidt (-no-romgs) since the basis
#   is mass-orthonormalized offline.
# - Single window: numWindows defaults to 0 and -nwinsamp is omitted.

set -e

P="-m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.02"   # single-window Sedov
RUN="srun"

# Full pipeline for a given sampling type and output directory.
# $1 = sampling type (eqp | eqp_energy), $2 = output dir, $3 = prep ranks
run_pipeline () {
  TYPE=$1; OUT=$2; NP=$3

  # 1. FOM offline (parallel): collect snapshots and write the FOM solution.
  $RUN -n $NP laghos -o $OUT $P -offline -no-romsns -no-romoffset \
       -rpar 0 -sample-stages -sdim 1000 -writesol

  # 2. Merge snapshots into single-window POD bases (no $P; no -nwinsamp).
  $RUN -n $NP ./merge -o $OUT -nset 1 -no-romsns -no-romoffset -eqp

  # 3. Online prep (parallel): basis enrichment + mass Gram-Schmidt for
  #    CEQP, plus the NNLS reduced quadrature rule.
  $RUN -n $NP laghos -o $OUT $P -online -romhrprep -no-romsns \
       -no-romoffset -no-romgs -hrsamptype $TYPE -maxnnls 100

  # 4. Online hyperreduced (serial: sample mesh is on rank 0).
  $RUN -n 1 laghos -o $OUT $P -online -romhr -no-romsns \
       -no-romoffset -no-romgs -hrsamptype $TYPE

  # 5. Restore: print relative errors of the ROM solution vs the FOM.
  $RUN -n $NP laghos -o $OUT $P -restore -soldiff -no-romsns \
       -no-romoffset -hrsamptype $TYPE
}

# Baseline basic EQP and our energy-conserving EQP, both with 2-rank prep.
run_pipeline eqp        sedov_eqp_2r   2
run_pipeline eqp_energy sedov_ceqp_2r  2

# Rank-invariance check for the distributed mass Gram-Schmidt:
# run CEQP again with serial prep and compare the restore errors.
run_pipeline eqp_energy sedov_ceqp_1r  1

# Compare:
# - basic vs CEQP: restore relative errors of sedov_eqp_2r vs sedov_ceqp_2r
#   should be the same order of magnitude (approximate agreement).
# - mass-GS rank invariance: sedov_ceqp_1r vs sedov_ceqp_2r restore errors
#   should match to round-off.
#
# Tunables: raise -maxnnls if the combined CEQP NNLS does not converge;
# add -lqnnls to stage 3 for the LQ preconditioning used in the paper.
