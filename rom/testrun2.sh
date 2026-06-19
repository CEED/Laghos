#!/bin/bash
# Multi-window Sedov validation for the energy-conserving EQP (CEQP)
# work, compared against the existing basic EQP method.
# The runs use the global IC as the offset variables for every window.
#
# Run from the ROM build directory (where laghos, merge and data/ live).
#
# Notes:
# - Stages 1, 2, 3, 5 run in parallel; stage 4 (online hyperreduced) runs
#   serially because the EQP sample mesh lives on rank 0.
# - The distributed mass Gram-Schmidt runs in stage 3 (romhrprep), so the
#   2-rank run exercises it. Comparing the 1-rank and 2-rank CEQP runs is
#   the rank-invariance check.
# - Offsets are kept ON (-romos): offline subtracts the initial state,
#   making the X/V/E snapshot counts consistent (without offsets the
#   nonzero initial position is sampled while the zero initial velocity
#   is skipped as trivial, breaking the merge snapshot-size check).
# - SNS is kept ON (-romsns) so merge skips the Fv/Fe snapshots that the
#   EQP offline does not write; it does not affect the EQP basis.
# - Runtime Gram-Schmidt is OFF (-no-romgs): the basis is
#   mass-orthonormalized offline.

set -e

P="-m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.04 -s 7"   # single-window Sedov
RUN="srun"

# Run the single window through the windowed code path as a one-window
# case (-nwin 1), rather than the non-windowed branch. The online phase
# has no built-in dimension resolution: without a time-window parameter
# (twp) file it leaves the ROM dimensions at their -1 default, which
# aborts in the CAROM::Vector ctor (dim > 0). The merge writes the twp
# file (the per-window energy-fraction dimensions) only when windowing is
# requested, so we force a single basis window at merge with a sample
# count far above the snapshot count (-nwinsamp 1000); the online stages
# then read the dimensions from the twp file with -nwin 1.

# Full pipeline for a given sampling type and output directory.
# $1 = sampling type (eqp | eqp_energy), $2 = output dir, $3 = prep ranks
run_pipeline () {
  TYPE=$1; OUT=$2; NP=$3

  # 1. FOM offline (parallel): collect snapshots and write the FOM solution.
  $RUN -n $NP laghos -o $OUT $P -offline -romsns -romos \
       -rpar 0 -sample-stages -sdim 1000 -writesol

  # 2. Merge snapshots into windows.
  $RUN -n $NP ./merge -o $OUT -nset 1 -romsns -romos -eqp -nwinsamp 10

  # 3. Online prep (parallel): basis enrichment + mass Gram-Schmidt for
  #    CEQP, plus the NNLS reduced quadrature rule.
  $RUN -n $NP laghos -o $OUT $P -online -romhrprep -romsns \
       -romos -no-romgs -nwin 5 -hrsamptype $TYPE -maxnnls 500

  # 4. Online hyperreduced (serial: sample mesh is on rank 0).
  $RUN -n 1 laghos -o $OUT $P -online -romhr -romsns \
       -romos -no-romgs -nwin 5 -hrsamptype $TYPE

  # 5. Restore: print relative errors of the ROM solution vs the FOM.
  $RUN -n $NP laghos -o $OUT $P -restore -soldiff -romsns \
       -romos -nwin 5 -hrsamptype $TYPE
}

# Baseline basic EQP and our energy-conserving EQP, both with 2-rank prep.
run_pipeline eqp        sedov_eqp_mw_2r   2
run_pipeline eqp_energy sedov_ceqp_mw_2r  2

# Rank-invariance check for the distributed mass Gram-Schmidt:
# run CEQP again with serial prep and compare the restore errors.
run_pipeline eqp_energy sedov_ceqp_mw_1r  1

# Compare:
# - basic vs CEQP: restore relative errors of sedov_eqp_2r vs sedov_ceqp_2r
#   should be the same order of magnitude (approximate agreement).
# - mass-GS rank invariance: sedov_ceqp_1r vs sedov_ceqp_2r restore errors
#   should match to round-off.
#
# Tunables: raise -maxnnls if the combined CEQP NNLS does not converge;
# add -lqnnls to stage 3 for the LQ preconditioning used in the paper.
