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

command="mpirun -np "$((ntask))" laghos"
comment="mpirun -np X laghos"

command1="mpirun -np 1 laghos"
comment1="mpirun -np 1 laghos"

# Taylor-Green 2D pure
params="-p 0 -dim 2 -s 7 -tf 0.25 -rs 3 -vs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Taylor-Green 2D mixed
params="-p 0 -dim 2 -s 7 -tf 0.25 -rs 3 -vs 1 -mm -s_v 0 -s_e 0"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Taylor-Green 2D shifted
params="-p 0 -dim 2 -s 7 -tf 0.05 -rs 3 -vs 1 -mm -s_v 1 -s_e 4"
run_line=$command1" "$params
com_line=$comment1" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Sod 1D pure
params="-p 8 -dim 1 -s 7 -z 100 -rs 0 -tf 0.1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Sod 1D mixed
params="-p 8 -dim 1 -s 7 -z 100 -rs 0 -tf 0.1 -mm -s_v 0 -s_e 0"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Sod 1D shifted
params="-p 8 -dim 1 -s 7 -z 100 -rs 0 -tf 0.02 -mm -s_v 1 -s_e 4"
run_line=$command1" "$params
com_line=$comment1" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Water-Air 1D pure
params="-p 9 -dim 1 -s 7 -z 100 -rs 0 -tf 1.0e-4"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Water-Air 1D pure mixed
params="-p 9 -dim 1 -s 7 -z 100 -rs 0 -tf 1.0e-4 -mm"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Water-Air 1D pure shifted
params="-p 9 -dim 1 -s 7 -z 100 -rs 0 -tf 1.0e-4 -mm -s_v 1 -s_e 4"
run_line=$command1" "$params
com_line=$comment1" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Triple Point 2D pure
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 1 -rs 1"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Triple Point 2D mixed
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 1 -rs 1 -mm -s_v 0 -s_e 0"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Triple Point 2D shifted
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 0.1 -rs 1 -mm -s_v 1 -s_e 4"
run_line=$command1" "$params
com_line=$comment1" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Triple Point 2D ALE
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 2.0 -rs 1 -mm -ale 0.5"
run_line=$command" "$params
com_line=$comment" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

# Triple Point 2D shifted ALE
params="-p 10 -m data/rectangle01_quad.mesh -s 7 -tf 0.2 -rs 1 -mm -ale 0.1 -s_v 1 -s_e 4"
run_line=$command1" "$params
com_line=$comment1" "$params
echo -e $com_line >> $file
$run_line | grep -e 'marker:' -e 'norm:' >> $file
echo -e >> $file

tkdiff $file autotest/baseline.out

cd autotest
exit 0
