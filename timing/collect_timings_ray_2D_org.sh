#!/usr/bin/env bash

#versions=('master' 'cpuL' 'cpuLT' 'gpuLT' 'gpuLT' 'raja' 'raja-cuda' 'occa' 'occa-cuda')
#          11       21    31      41       51     61     71          81      91
#versions=('gpuLT' 'gpuKT' 'raja-cuda' 'occa-cuda' 'gpuKT-uvm' 'gpuKT-share')
#          11      21      31         41           51          61
versions=('gpuKT-share')

MN=10
host=ray
maxsteps=10
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

#########################################################################
bkill -u camier1

#########################################################################
n=0
rmax=4 # must match {1..rmax} of sref
for problem in {1..1}; do
    outfile=timings_$host""_2d_p$problem""_ms$maxsteps.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|H1order|refs|H1dofs|L2dofs|H1cg_rate|L2cg_rate|forces_rate|update_quad_rate|total_time|total_rate|\n|" > $outfile
    for torder in 8; do
        for sref in 3; do
#    for torder in {9..9}; do
#        for sref in {4..4}; do
            for version in "${versions[@]}"; do
                ((n+=1))
#                echo $n
#                if [ $(( $n % 15 )) -eq 0 ] ; then
#                    until [ bjobs 2>1 | grep -m 1 "No unfinished job found"] ; do sleep 0.125 ; done
#                fi
                #sleep 1;
                #echo \#$version >> $outfile
                # Test for specific CUDA versions
                #echo version is $version
                version_name=$version
                additional_options=
                if [ $version == 'raja-cuda' ]; then
                    version_name=raja
                    additional_options=-cuda
                fi
                if [ $version == 'occa-cuda' ]; then
                    version_name=occa
                    additional_options="--occa-config cuda.json"
                fi
                if [ $version == 'gpuKT-uvm' ]; then
                    version_name=gpuKT
                    additional_options="-uvm"
                fi
                if [ $version == 'gpuKT-share' ]; then
                    version_name=gpuKT
                    additional_options="-share"
                fi
                echo -e "\e[35mlaghos-\e[32;1m$version\e[m\e[35m Q$((torder+1))Q$torder $sref"ref"\e[m"
#                bsub -J rLaghos -n 1 -I -x -W $MN -R "span[ptile=4]" mpirun ../laghos.$version_name \
#                    -p $problem -tf 0.5 -cfl 0.01 -vs 1 \
#                    --cg-tol 0 --cg-max-steps 50 \
#                    --max-steps $maxsteps \
#                   --mesh $mesh_file \
#                    --refine-serial $sref \
#                    --order-thermo $torder \
#                    --order-kinematic $((torder+1)) \
#                    $additional_options
                echo -n $(run_case bsub -n 1 -W $MN -I -x -R "span[ptile=4]" mpirun ../laghos.$version_name \
                    -p $problem -tf 0.5 -cfl 0.01 -vs 1 \
                    --cg-tol 0 --cg-max-steps 50 \
                    --max-steps $maxsteps \
                    --mesh $mesh_file \
                    --refine-serial $sref \
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
