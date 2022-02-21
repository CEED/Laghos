#!/bin/bash

# execute with
# ./test.sh n   -- to run on n tasks, getting test_n.out

ntask=$1
if [ $ntask = 0 ]; then
  ntask=1;
fi

cd ..
file="autotest/run_"$((ntask))".out"
rm -f $file

command="mpirun -np "$((ntask))" laghos"
comment="mpirun -np X laghos"

# Taylor-Green 2D
params="-p 0 -dim 2 -s 7 -tf 0.5 -rs 3 -vs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'Material marker' -e 'Face marker' >> $file
echo -e >> $file

# Sod 1D
params="-p 8 -dim 1 -s 7 -z 100 -rs 0 -tf 0.2"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'Material marker' -e 'Face marker' >> $file
echo -e >> $file

# Triple Point 2D
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 3.0 -rs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'Material marker' -e 'Face marker' >> $file
echo -e >> $file

tkdiff $file autotest/baseline.out

cd autotest
exit 0
