#!/usr/bin/env bash

# User input.
# To get Cartesian mesh partitions:
#   with the 222 partition, use 4/32/256/2048/16384 nodes.
#   with the 432 partition, use 12/96/768/6144(quarter machine) nodes.
#   with the 322 partition, use 6/48/384/3072/24576(full machine) nodes.
part_type=322
nodes=24576
l2orders=(1 2 3)
minL2dof_node=0
maxL2dof_node=200000
steps=2
cg_iter=50
# End of user input.

if (( part_type == 222 )); then
  nzones0=8
  mesh_file=../data/cube01_hex.mesh
elif (( part_type == 432 )); then
  nzones0=24
  mesh_file=../data/cube_24_hex.mesh
elif ((part_type == 322 )); then
  nzones0=12
  mesh_file=../data/cube_12_hex.mesh
fi

# Make sure that the serial mesh has at least one zone per task.
nproc=$(( 16 * nodes ))
sref=0
while (( nzones0 * 8**(sref) < nproc ))
do
  sref=$(( sref+1 ))
done
echo "sref: "$sref "serial_nzones: "$(( nzones0 * 8**(sref) )) "nproc: "$nproc

minL2dof=$(( minL2dof_node * nodes ))
maxL2dof=$(( maxL2dof_node * nodes ))

options=( 'pa' 'fa' )
outfile=timings_3d

run_case()
{
    # Pass command as all inputs
    # Outputs: order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate total_rate

    "$@" | tee -a run.log | awk '
BEGIN { ref = 0 }
/--refine-serial/ { ref += $2 }
/--refine-parallel/ { ref += $2 }
/--order/ { order = $2 }
/Number of kinematic/ { h1_dofs = $7 }
/Number of specific internal energy/ { l2_dofs = $7 }
/CG \(H1\) rate/ { h1_cg_rate = $9 }
/CG \(L2\) rate/ { l2_cg_rate = $9 }
/Forces rate/ { forces_rate = $8 }
/UpdateQuadData rate/ { update_quad_rate = $8 }
/Major kernels total rate/ { total_rate = $11 }
END { printf("%d %d %d %d %.8f %.8f %.8f %.8f %.8f\n", order, ref, h1_dofs, l2_dofs, h1_cg_rate, l2_cg_rate, forces_rate, update_quad_rate, total_rate) }'
}

for method in "${options[@]}"; do
  echo "# order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate total_rate" > $outfile"_"$method
  for torder in ${l2orders[@]}; do
    for pref in {0..10}; do
       nzones=$(( 8**(pref+sref)*nzones0 ))
       nL2dof=$(( nzones*(torder+1)**3 ))
       if (( nproc <= nzones )) && (( nL2dof > minL2dof )) && (( nL2dof < maxL2dof )) ; then
         echo "np"$nproc "Q"$((torder+1))"Q"$torder $pref"ref" $method $outfile"_"$method
# bigger runs need srun --reservation=...
#         echo $(run_case srun -n $nproc --reservation=ascasc-dat ../laghos -$method -p 1 -tf 0.8 -pt $part_type \
         echo $(run_case srun -n $nproc ../laghos -$method -p 1 -tf 0.8 -pt $part_type \
                       --cg-tol 0 --cg-max-steps $cg_iter \
                       --max-steps $steps \
                       --mesh $mesh_file \
                       --refine-serial $sref --refine-parallel $pref \
                       --order-thermo $torder \
                       --order-kinematic $((torder+1))) >> $outfile"_"$method
      fi
    done
  done
done
