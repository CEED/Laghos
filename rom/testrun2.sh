#!/bin/bash
# Multi-window Sedov validation for the energy-conserving EQP (CEQP)
# work, compared against the existing basic EQP method.
#
# CEQP (eqp_energy) now uses WINDOW-DEPENDENT offsets: the first sample
# of each window is the offset, absorbed as an extra column in the
# derived reduced bases.
# The offline phase still samples deviations (-romos) for an accurate
# per-window POD, but the online/restore phases run OFFSET-FREE
# (-no-romoffset): the absorbed columns represent the full initial
# fields, and the velocity offset vanishes structurally (v_os = 0), so
# conservation holds even for nonzero initial velocity.
# The basic EQP baseline uses the SAME per-window first-sample offset
# (-romos -rostype load everywhere), for a controlled comparison: both
# methods see the same offset, differing only in how it is handled
# (BEQP lifts it online; CEQP absorbs it into the basis to force v_os=0).
#
# Run from the ROM build directory (where laghos, merge and data/ live).
#
# Notes:
# - Stages 1, 2, 3, 5 run in parallel; stage 4 (online hyperreduced) runs
#   serially because the EQP sample mesh lives on rank 0.
# - The distributed mass Gram-Schmidt runs in stage 3 (romhrprep), so the
#   2-rank run exercises it. Comparing the 1-rank and 2-rank CEQP runs is
#   the rank-invariance check.
# - Offsets are kept ON offline (-romos): the offline phase subtracts the
#   (per-window) initial state, making the X/V/E snapshot counts
#   consistent (without offsets the nonzero initial position is sampled
#   while the zero initial velocity is skipped as trivial, breaking the
#   merge snapshot-size check) and giving an accurate deviation POD.
# - For CEQP, -rostype load selects the per-window first-sample offset
#   (saveLoadOffset); the offline+merge stages keep -romos (useOffset=1,
#   matching the offline_param record), while the online/restore stages
#   pass -no-romoffset to run offset-free with the absorbed columns.
# - SNS is kept ON (-romsns) so merge skips the Fv/Fe snapshots that the
#   EQP offline does not write; it does not affect the EQP basis.
# - Runtime Gram-Schmidt is OFF (-no-romgs): the basis is
#   mass-orthonormalized offline.

set -e

P="-m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.08 -s 7"   # multi-window Sedov
RUN="srun"

# Four ROM time windows, resolved from the twp file written by merge.
# The merge needs windowing requested to write the twp (per-window
# energy-fraction dimensions); a sample count above the snapshot count
# (-nwinsamp 10 here) sets the window boundaries, and the online stages
# read the dimensions back with -nwin 4 (no explicit -rdim* needed).

# Full pipeline for a given sampling type and output directory.
# $1 = sampling type (eqp | eqp_energy), $2 = output dir, $3 = prep ranks
run_pipeline () {
  TYPE=$1; OUT=$2; NP=$3

  if [ "$TYPE" = "eqp_energy" ]; then
    # CEQP: window-dependent offsets absorbed as basis columns.
    # Offline + merge sample deviations with offsets on (useOffset = 1);
    # online + restore run offset-free with the absorbed columns.
    OS_OFF="-romos -rostype load"        # offline and merge
    OS_ON="-no-romoffset -rostype load"  # online prep / online hr / restore
  else
    # Basic EQP baseline: SAME per-window first-sample offset as CEQP, for
    # a controlled comparison. Basic EQP has no zero-offset constraint and
    # no offline/online dichotomy, so it keeps offsets on everywhere
    # (-romos -rostype load) and lifts the per-window offset online.
    OS_OFF="-romos -rostype load"
    OS_ON="-romos -rostype load"
  fi

  # 1. FOM offline (parallel): collect snapshots and write the FOM solution.
  $RUN -n $NP laghos -o $OUT $P -offline -romsns $OS_OFF \
       -rpar 0 -sample-stages -sdim 1000 -writesol

  # 2. Merge snapshots into windows (offsets on to match the record).
  $RUN -n $NP ./merge -o $OUT -nset 1 -romsns $OS_OFF -eqp -nwinsamp 10

  # 3. Online prep (parallel): basis enrichment + offset-column absorption
  #    + mass Gram-Schmidt for CEQP, plus the NNLS reduced quadrature rule.
  $RUN -n $NP laghos -o $OUT $P -online -romhrprep -romsns \
       $OS_ON -no-romgs -nwin 4 -hrsamptype $TYPE -lqnnls -maxnnls 500

  # 4. Online hyperreduced (serial: sample mesh is on rank 0).
  $RUN -n 1 laghos -o $OUT $P -online -romhr -romsns \
       $OS_ON -no-romgs -nwin 4 -hrsamptype $TYPE

  # 5. Restore: print relative errors of the ROM solution vs the FOM.
  $RUN -n $NP laghos -o $OUT $P -restore -soldiff -romsns \
       $OS_ON -nwin 4 -hrsamptype $TYPE
}

# Baseline basic EQP and our energy-conserving EQP, both with 2-rank prep.
run_pipeline eqp        sedov_eqp_mw_2r   2
run_pipeline eqp_energy sedov_ceqp_mw_2r  2

# Rank-invariance check for the distributed mass Gram-Schmidt:
# run CEQP again with serial prep and compare the restore errors.
run_pipeline eqp_energy sedov_ceqp_mw_1r  1

# Compare:
# - basic vs CEQP: restore relative errors of sedov_eqp_mw_2r vs
#   sedov_ceqp_mw_2r should be the same order of magnitude (approximate
#   agreement). Both now use the same per-window offset, so this is a
#   controlled accuracy comparison (CEQP additionally conserves energy).
# - mass-GS rank invariance: sedov_ceqp_mw_1r vs sedov_ceqp_mw_2r restore
#   errors should match to round-off.
#
# Tunables: raise -maxnnls if the combined CEQP NNLS does not converge;
# add -lqnnls to stage 3 for the LQ preconditioning used in the paper.
