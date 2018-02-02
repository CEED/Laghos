#!/usr/bin/env bash
#           10      20        30     40          50     60
#versions=( 'master' 'kernels' 'raja' 'raja-cuda' 'occa' 'occa-cuda')
#versions=( 'master' 'kernels' 'raja' 'occa')
#versions=('kernels' 'kernels-share' 'kernels-cuda' 'kernels-cuda-share')
#versions=( 'master' 'raja-cuda' 'occa-cuda')
versions=('master' 'cpuL' 'cpuLT' 'gpuLT' 'gpuLT' 'raja' 'raja-cuda' 'occa' 'occa-cuda')
#          11       21    31      41       51     61     71          81      91

host=tux
maxsteps=1
parallel_refs=0
mesh_file=../data/square01_quad.mesh

#########################################################################
run_case() {
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
END { printf("%d|%d|%d|%d|%.8f|%.8f|%.8f|%.8f|%.8f|%.8f|\n",
   order,ref,h1_dofs,l2_dofs,
   h1_cg_rate,l2_cg_rate,forces_rate,update_quad_rate,
   total_time, total_rate) }'
}

#bsub -Ip -x -G guests mpirun -n 1
#########################################################################
rmax=6 # must match {1..rmax} of sref
for problem in {0..1}; do
    outfile=timings_$host""_2d_p$problem""_ms$maxsteps.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|H1order|refs|H1dofs|L2dofs|H1cg_rate|L2cg_rate|forces_rate|update_quad_rate|total_time|total_rate|\n|" > $outfile
    for torder in {0..5}; do
        for sref in {1..6}; do
            for version in "${versions[@]}"; do
                #echo \#$version >> $outfile
                # Test for specific CUDA versions
                #echo version is $version
                version_name=$version
                additional_options=
                if [ $version == 'kernels-share' ]; then
                    version_name=kernels
                    additional_options=-share
                fi
                if [ $version == 'kernels-cuda' ]; then
                    version_name=kernels
                    additional_options=-cuda
                fi
                if [ $version == 'kernels-cuda-share' ]; then
                    version_name=kernels
                    additional_options="-cuda -share"
                fi
                if [ $version == 'raja-cuda' ]; then
                    version_name=raja
                    additional_options=-cuda
                fi
                if [ $version == 'occa-cuda' ]; then
                    version_name=occa
                    additional_options="--occa-config cuda.json"
                fi
                echo -e "\e[35mlaghos-\e[32;1m$version\e[m\e[35m Q$((torder+1))Q$torder $sref"ref"\e[m"
                echo ../laghos.$version_name \
                    -p $problem -tf 0.5 -cfl 0.05 -vs 1 \
                    --cg-tol 0 --cg-max-steps 50 \
                    --max-steps $maxsteps \
                    --mesh $mesh_file \
                    --refine-serial $sref \
                    --refine-parallel $parallel_refs \
                    --order-thermo $torder \
                    --order-kinematic $((torder+1)) \
                    $additional_options
                echo -n $(run_case ../laghos.$version_name \
                    -p $problem -tf 0.5 -cfl 0.05 -vs 1 \
                    --cg-tol 0 --cg-max-steps 50 \
                    --max-steps $maxsteps \
                    --mesh $mesh_file \
                    --refine-serial $sref \
                    --refine-parallel $parallel_refs \
                    --order-thermo $torder \
                    --order-kinematic $((torder+1)) \
                    $additional_options)>> $outfile
            done
            if [ $((sref)) != $rmax ]; then 
                echo -ne "\n|">> $outfile
            else
                echo -ne "\n\n|">> $outfile
            fi
        done
    done
done
