#!/usr/bin/env bash
versions=('gpuKT' 'gpuKT-hcpo')

MN=5
host=ray
maxsteps=10
mesh_file=../data/cube01_hex.mesh

#########################################################################
n=0
rmax=6 # must match {1..rmax} of sref
for problem in {1..1}; do
    outfile=timings_$host""_3d_p$problem""_ms$maxsteps""_mult.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|H1order|refs|H1dofs|L2dofs|H1cg_rate|L2cg_rate|forces_rate|update_quad_rate|total_time|total_rate|\n|" > $outfile
    for torder in {1..9}; do
        for sref in {1..6}; do
            for version in "${versions[@]}"; do
                if [ $(( $n % 4 )) -eq 0 ] ; then
                    until bjobs 2>&1 | grep -m 1 "No unfinished job found"; do : ; done
                fi
                version_name=$version
                additional_options=
                if [ $version == 'gpuKT-hcpo' ]; then
                    version_name=gpuKT
                    additional_options="-hcpo"
                fi
                echo -e "\e[35mlaghos-\e[32;1m$version\e[m\e[35m Q$((torder+1))Q$torder $sref"ref"\e[m"
                jobn=$version""_p$problem""_ms$maxsteps""_q$torder""_r$sref
                if [ -f $jobn.out ]; then continue; fi
                ((n+=1))
                bsub -x -J $jobn -o $jobn.out \
                    -q pbatch -n 8 -W $MN -R "span[ptile=4]" \
                    mpirun -gpu ../laghos.$version_name -aware \
                    -mult --max-steps 4 \
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
