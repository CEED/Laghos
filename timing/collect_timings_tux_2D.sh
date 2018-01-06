#!/usr/bin/env bash

versions=( 'master' 'kernels' 'raja' 'raja-cuda' 'occa' 'occa-cuda')
#versions=( 'occa' 'occa-cuda')

problem=0
parallel_refs=0
maxL2dof=1000000

maxsteps=1

outfile=timings_tux_2d
mesh_file=../data/square01_quad.mesh

calc() { awk "BEGIN{print $*}"; }

run_case()
{
    # Pass command as all inputs
    # Outputs: order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate

    "$@" | awk '
BEGIN { ref= 0 }
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
/Major kernels total time/ { total_time = $6 }
END { printf("%d %d %d %d %.8f %.8f %.8f %.8f %.8f %.8f\n", order, ref, h1_dofs, l2_dofs, h1_cg_rate, l2_cg_rate, forces_rate, update_quad_rate, total_time, total_rate) }'
}

[ -r $outfile ] && cp $outfile $outfile.bak
for version in "${versions[@]}"; do
    echo "# H1order refs h1_dofs l2_dofs h1_cg_rate l2_cg_rate forces_rate update_quad_rate total_time total_rate" > $outfile"_"$version
    # Test for specific CUDA versions
    #echo version is $version
    version_name=$version
    additional_options=
    if [ $version == 'raja-cuda' ]; then
        #echo RAJA CUDA
        #version=raja
        version_name=raja
        additional_options=-cuda
    fi
    if [ $version == 'occa-cuda' ]; then
        #echo OCCA CUDA
        #version=occa
        #version_extra="-cuda"
        version_name=occa
        #additional_options="--device-info \"mode:'CUDA',deviceID:0\""
        additional_options="--occa-config cuda.json"
    fi
    for torder in {1..5}; do
        for sref in {6..6}; do
            nzones=$(( 4**(sref+1) ))
            nL2dof=$(( nzones*(torder+1)**2 ))
            if (( nproc <= nzones )) && (( nL2dof < maxL2dof )) ; then
                echo -e \\e[35mlaghos-\\e[32\;1m$version\\e[m\\e[35m Q$((torder+1))Q$torder $sref"ref"\\e[m
#                ../laghos-$version_name \
#                    -p $problem -tf 0.5 -cfl 0.05 -vs 1 \
#                    --cg-tol 0 --cg-max-steps 50 \
#                    --max-steps $maxsteps \
#                    --mesh $mesh_file \
#                    --refine-serial $sref \
#                    --refine-parallel $parallel_refs \
#                    --order-thermo $torder \
#                    --order-kinematic $((torder+1)) \
#                    $additional_options
                echo $(run_case ../laghos-$version_name \
                    -p $problem -tf 0.5 -cfl 0.05 -vs 1 \
                    --cg-tol 0 --cg-max-steps 50 \
                    --max-steps $maxsteps \
                    --mesh $mesh_file \
                   --refine-serial $sref \
                    --refine-parallel $parallel_refs \
                    --order-thermo $torder \
                    --order-kinematic $((torder+1)) \
                    $additional_options)>> $outfile"_"$version
            fi
        done
    done
done
