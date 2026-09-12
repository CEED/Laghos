#!/bin/bash
# Summarize the BEQP/CEQP test outputs in one table, one row per run dir.
# Each test script's stdout is expected in <name>.out next to this script.
# Usage: summary.sh [file.out ...]  (default: all .out files here)
#
# Columns:
#   nwin       number of time windows (from the online prep stage)
#   E0         initial total energy of the FOM
#   FOM dE     FOM relative energy difference (offline stage)
#   online dE  reduced-energy diagnostic (CEQP online stage only)
#   restore dE relative energy difference of the restored ROM solution
#   err X/V/E  restore relative errors of position, velocity, energy

DIR=$(dirname "$0")
if [ $# -gt 0 ]; then FILES="$@"; else FILES="$DIR"/*.out; fi

awk '
# Every laghos or merge run starts with an options block.
/^Options used:/ { stage = ""; hr = 0; prep = 0 }
/^   --outputfilename / {
  run = $2
  if (!(run in seen)) { seen[run] = 1; order[++n] = run }
}
/^   --offline$/    { stage = "offline" }
/^   --online$/     { stage = "online" }
/^   --restore$/    { stage = "restore" }
/^   --romhr$/      { hr = 1 }
/^   --romhrprep$/  { prep = 1 }
# Count the windows once, in the hyper-reduction prep stage.
/^Using time window / { if (stage == "online" && prep) nwin[run]++ }
/^Initial energy: / { if (stage == "offline") e0[run] = $3 }
/^Rel\. energy diff: / {
  if (stage == "offline") fom[run] = $4
  else if (stage == "online" && hr) onl[run] = $4
  else if (stage == "restore") res[run] = $4
}
/Sol_Position Rel\. DIFF norm / { ex[run] = $NF }
/Sol_Velocity Rel\. DIFF norm / { ev[run] = $NF }
/Sol_Energy Rel\. DIFF norm /   { ee[run] = $NF }
function val(x) { return (x == "") ? "-" : x }
END {
  fmt = "%-20s %4s %12s %12s %12s %12s %12s %12s %12s\n"
  printf fmt, "run", "nwin", "E0", "FOM dE", "online dE",
    "restore dE", "err X", "err V", "err E"
  for (i = 1; i <= n; i++) {
    r = order[i]
    printf fmt, r, val(nwin[r]), val(e0[r]), val(fom[r]), val(onl[r]),
      val(res[r]), val(ex[r]), val(ev[r]), val(ee[r])
  }
}
' $FILES
