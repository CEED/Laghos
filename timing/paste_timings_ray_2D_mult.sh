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
# grep killed *.out
#########################################y################################
rmax=12 # must match {1..rmax} of sref
for problem in {1..1}; do
    outfile=timings_$host""_2d_p$problem""_ms$maxsteps""_mult.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|order|refs|gdofs|ldofs|elapsed|\n|" > $outfile
    for torder in {1..9}; do
        for sref in {1..12}; do
            for version in "${versions[@]}"; do
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
                cat $jobn.out |grep -v mpirun | awk '
BEGIN { ref= 0 }
/--refine-serial/ { ref += $2 }
/--refine-parallel/ { ref += $2 }
/--order/ { order = $2 }
/Number of global dofs/ { gdofs = $5 }
/Number of local dofs/ { ldofs = $5 }
/Elapsed/ { elapsed = $5 }
END { printf("%d|%d|%d|%d|%.8f|",
   order,ref,gdofs,ldofs,elapsed) }' >> $outfile
            done
            if [ $((sref)) != $rmax ]; then 
                echo -ne "\n|">> $outfile
            else
                echo -ne "\n\n|">> $outfile
            fi
        done
    done
done
