#!/usr/bin/env bash

#versions=('master' 'cpuL' 'cpuLT' 'gpuLT' 'gpuLT' 'raja' 'raja-cuda' 'occa' 'occa-cuda')
#          11       21    31      41       51     61     71          81      91
#versions=('gpuLT' 'gpuKT' 'raja-cuda' 'occa-cuda' 'gpuKT-uvm' 'gpuKT-share')
#          11      21      31         41           51          61
#versions=('occa-cuda' 'raja-cuda' 'gpuKT-share')
versions=('gpuKT' 'gpuKT-hcpo')

MN=10
host=ray
maxsteps=10
mesh_file=../data/quad.mesh

#########################################################################
n=0
rmax=12 # must match {1..rmax} of sref
for problem in {1..1}; do
    outfile=timings_$host""_2d_p$problem""_ms$maxsteps""_mult.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|H1order|refs|H1dofs|L2dofs|H1cg_rate|L2cg_rate|forces_rate|update_quad_rate|total_time|total_rate|\n|" > $outfile
    for torder in {1..9}; do
        for sref in {1..12}; do
            for version in "${versions[@]}"; do
                if [ $(( $n % 4 )) -eq 0 ] ; then
                    until bjobs 2>&1 | grep -m 1 "No unfinished job found"; do : ; done
                fi
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
                if [ $version == 'gpuKT-hcpo' ]; then
                    version_name=gpuKT
                    additional_options="-hcpo"
                fi
                echo -e "\e[35mlaghos-\e[32;1m$version\e[m\e[35m Q$((torder+1))Q$torder $sref"ref"\e[m"
                jobn=$version""_p$problem""_ms$maxsteps""_q$torder""_r$sref
                if [ -f $jobn.out ]; then continue; fi
                ((n+=1))
                bsub -x -J $jobn -o $jobn.out \
                    -q pbatch -n 4 -W $MN -R "span[ptile=4]" \
                    mpirun -gpu ../laghos.$version_name -aware \
                    -mult --max-steps $maxsteps \
                    --mesh $mesh_file --refine-serial $sref \
                    --order-thermo $torder --order-kinematic $((torder+1)) \
                    $additional_options
            done
            if [ $((sref)) != $rmax ]; then 
                echo -ne "\n|">> $outfile
            else
                echo -ne "\n\n|">> $outfile
            fi
        done
    done
done
