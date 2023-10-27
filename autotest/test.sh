#!/bin/bash

# execute with
# ./test.sh n   -- to run on n tasks, getting test_n.out

if [ "$1" = "" ]; then
  ntask=1;
else
  ntask=$1;
fi

if [ $ntask = 0 ]; then
  ntask=1;
fi

cd ..
file="autotest/run_"$((ntask))".out"
rm -f $file

command="mpirun -np "$((ntask))" laghos -p 1 -s 7 -vs 50 -penPar 10.0 -per 12.0"
comment="mpirun -np X laghos"


# 2D trapezoid.
params="-m data/trapezoid_quad.mesh -tf 1.5 -rs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line >> $file
echo -e >> $file

# 2D circular hole.
params="-m data/refined.mesh -tf 0.4 -rs 2"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line >> $file
echo -e >> $file

# 2D circular hole Q3Q2.
params="-m data/refined.mesh -tf 0.8 -rs 1 -ok 3 -ot 2"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line >> $file
echo -e >> $file

# 2D disc.
params="-m data/disc-nurbs.mesh -tf 5.0 -rs 2"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line >> $file
echo -e >> $file

# 3D cube.
params="-m data/cube01_hex.mesh -tf 0.25 -rs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line >> $file
echo -e >> $file

tkdiff autotest/baseline.out $file

cd autotest
exit 0
