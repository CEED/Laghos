#!/usr/bin/env bash
versions=('gpuKT' 'gpuKT-hcpo')

MN=5
host=ray
maxsteps=10
mesh_file=../data/cube01_hex.mesh

#########################################################################
# grep killed *.out
#########################################y################################
rmax=6 # must match {1..rmax} of sref
for problem in {1..1}; do
    outfile=timings_$host""_3d_p$problem""_ms$maxsteps""_mult.org
    [ -r $outfile ] && cp $outfile $outfile.bak
    echo -ne "|order|refs|gdofs|ldofs|elapsed|\n|" > $outfile
    for torder in {1..9}; do
        for sref in {1..6}; do
            for version in "${versions[@]}"; do
                version_name=$version
                additional_options=
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
