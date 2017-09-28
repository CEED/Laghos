#!/usr/bin/env bash

options=( 'pa' 'fa' )

parallel_refs=0
nproc=4
mesh_dim=3
maxL2dof=200000

outfile=timings

if (( mesh_dim == 2 )); then
  mesh_file=data/square01_quad.mesh
else
  mesh_file=data/cube01_hex.mesh
fi

calc() { awk "BEGIN{print $*}"; }

run_case()
{
    # Pass command as all inputs
    # Outputs: order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate

    "$@" | tee run.log | awk '
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
END { printf("%d %d %d %d %.8f %.8f %.8f %.8f\n", order, ref, h1_dofs, l2_dofs, h1_cg_rate, l2_cg_rate, forces_rate, update_quad_rate) }'
}

[ -r $outfile ] && cp $outfile $outfile.bak
echo "# order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate" > $outfile"_"${options[0]}
echo "# order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate" > $outfile"_"${options[1]}
for method in "${options[@]}"; do
  for torder in {1..7}; do
    for sref in {0..10}; do
       nzones=$(( (2**mesh_dim)**(sref+1) ))
       nL2dof=$(( nzones*(torder+1)**mesh_dim ))
       if (( nproc <= nzones )) && (( nL2dof < maxL2dof )) ; then
         echo "np"$nproc "Q"$((torder+1))"Q"$torder $sref"ref" $method
         echo $(run_case mpirun -np $nproc ./laghos -$method \
                       -p 0 -tf 0.5 -cfl 0.05 -vs 1 \
                       --max_steps 10 \
                       --mesh $mesh_file \
                       --refine-serial $sref \
                       --refine-parallel $parallel_refs \
                       --order-thermo $torder \
                       --order-kinematic $((torder+1))) >> $outfile"_"$method
      fi
    done
  done
done
