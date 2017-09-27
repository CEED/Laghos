#!/usr/bin/env bash

# Orders to run
thermo_orders=( 1 2 3 4 )

# Number of CPUs to use
serial_refs=( 1 2 3 4 5 )

options=( 'pa' 'fa' )

parallel_refs=0

nproc=4

# Lower and upper limits on number of thermodynamic DOFs per processor
# The cases execute when DOFs calls within these limits
lower_dof_limit=50
upper_dof_limit=5000

outfile=timings

mesh_file2D=data/square01_quad.mesh
mesh_file3D=data/cube01_hex.mesh
mesh_dim=2

calc() { awk "BEGIN{print $*}"; }

run_case()
{
    # Pass command as all inputs
    # Outputs: order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate

    "$@" | awk '
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
  for torder in "${thermo_orders[@]}"; do
    for sref in "${serial_refs[@]}"; do
       echo "np"$nproc "Q"$((torder+1))"Q"$torder $sref"ref" $method
       echo $(run_case mpirun -np $nproc ./laghos -$method \
                       -p 1 -tf 0.8 -cfl 0.05 \
                       --max_steps 10 \
                       --mesh $mesh_file2D \
                       --refine-serial $sref \
                       --refine-parallel $parallel_refs \
                       --order-thermo $torder \
                       --order-kinematic $((torder+1))) >> $outfile"_"$method
    done
  done
done
