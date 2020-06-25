###############################################################################
# SET UP
###############################################################################

# Get options
while getopts ":i:e:n:" o;
do
	case "${o}" in
		i)
			i=${OPTARG}
      ;;
    e)
      e=${OPTARG}
      ;;
		n)
			n=${OPTARG}
			;;
    *)
      usage
      ;;
    esac
done
shift $((OPTIND-1))

# If both include and exclude are set, fail
if [ -n "${i}" ] && [ -n "${e}" ]; then
    usage
		exit 1
fi

# If normtypes option is set, set normtypes
normtypes=( $n )
if [ ${#normtypes[@]} -eq 0 ]; then
		normtypes=("l2")
fi

# Save directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Accumulate tests to run
testsToRun=( $i )
testsToExclude=( $e )
if [ ${#testsToRun[@]} -eq 0 ]
then
	echo "Running all tests"
	for file in $DIR/*;
	do
		fileName="$(basename "$file")"

		# Exclude tests from user option
		if [[ " ${testsToExclude[@]} " =~ " ${fileName} " ]]; then
			echo "${file} skipped"
			continue 1
		fi

		# Add test
    if [[ -d "$file" ]] && [[ $fileName != "Laghos" ]] && [[ $fileName != "results" ]]; then
			testsToRun+=($fileName)
    fi
	done
fi

###############################################################################
# COMPILATION
###############################################################################

# Compile the C++ comparators
g++ -std=c++11 -o $DIR/fileComparator $DIR/fileComparator.cpp
g++ -std=c++11 -o $DIR/basisComparator $DIR/basisComparator.cpp \
-I$DIR/../../libROM -L$DIR/../../libROM/build -lROM -Wl,-rpath,$DIR/../../libROM/build

# Clone and compile rom-dev branch of Laghos
git -C $DIR clone -b rom-dev https://github.com/CEED/Laghos.git

# Copy user.mk
cp $DIR/../user.mk $DIR/Laghos/user.mk

# Save directory of the baseline Laghos executable
BASELINE_LAGHOS_DIR=$DIR/Laghos

# Check that rom-dev branch of Laghos is present
if [ ! -d $BASE_LAGHOS_DIR ]; then
	echo "Baseline Laghos directory could not be cloned"
	exit 1
fi

# Save directory of the new Laghos executable
BASE_DIR=$DIR/..

# Build the baseline Laghos executable
make --directory=$BASELINE_LAGHOS_DIR CURDIR="$BASE_DIR"

# Check if make built correctly
if [ $? -ne 0 ]
then
  exit 1
fi

# Create results directory
RESULTS_DIR=$DIR/results
if [ ! -d $RESULTS_DIR ]; then
  mkdir -p $RESULTS_DIR;
else
  rm -rf $RESULTS_DIR/*
fi

###############################################################################
# RUN TESTS
###############################################################################

# Test type
testtypes=(offline online romhr restore)

# Run all tests
for simulation in "${testsToRun[@]}"
do

	# Check if simulation directory exists
	if [ ! -d $DIR/$simulation ]; then
		echo "$simulation test directory doesn't exist"
		exit 1
	fi

  # Create simulation results log file
  simulationLogFile=${RESULTS_DIR}/${simulation}.txt
  touch $simulationLogFile

	# Run every script in each test directory
	for script in $DIR/$simulation/*;
	do

		if [[ "$script" == *".sh" ]]; then

      # Get script name without extension
			scriptName=$(basename $script)
      scriptName="${scriptName%.*}"

			# Run script with the following normtypes
			for normtype in "${normtypes[@]}"
			do

        # Clear run directories
        echo "Cleaning run directories"
        make --directory=$BASE_DIR CURDIR="$BASE_DIR" clean-exec
        make --directory=$BASELINE_LAGHOS_DIR CURDIR="$BASE_DIR" clean-exec

				echo "Running $scriptName with normtype: $normtype"

        # Run every test type
        for testtype in "${testtypes[@]}"
        do

  				# Run simulation from rom-dev branch
  			  echo $"Running baseline simulation for comparison"
  				(cd $BASELINE_LAGHOS_DIR && . "$script")

					if [ "$?" -ne 0 ]
					then
						echo "Something went wrong running the baseline simulation with the
            test script: $scriptName" | tee -a $simulationLogFile
            echo "${scriptName}_${testtype}_${normtype}: FAIL" | tee -a $simulationLogFile

            # Skip to next test script
						break 2
					fi

  				# # Run simulation from current branch
  				echo $"Running new simulation for regression testing"
  				(cd $BASE_DIR && . "$script")

          if [ "$?" -ne 0 ]
					then
						echo "Something went wrong running the new user branch simulation with the
             test script: $scriptName" | tee -a $simulationLogFile
            echo "${scriptName}_${testtype}_${normtype}: FAIL" | tee -a $simulationLogFile

            # Skip to next test script
						break 2
					fi

          # Find number of steps simulation took in rom-dev to compare final timestep later
          num_steps=$(head -n 1 $BASELINE_LAGHOS_DIR/run/num_steps)

  				# After simulations complete, compare results
  				for testFile in $BASELINE_LAGHOS_DIR/run/*
  				do
  					fileName="$(basename "$testFile")"

            # Skip if not correct file type to compare
            check_file_type() {
              case $testtype in
                offline)
                  if [[ "$fileName" != "basis"* ]] && [[ "$fileName" != "Sol"* ]] &&
                  [[ "$fileName" != "sVal"* ]] && [[ "$fileName" != "num_steps" ]]; then
                    continue 1
                  fi
                  ;;
                online | romhr)
                  if [[ "$fileName" != *"norm"* ]] && [[ "$fileName" != "ROMsol" ]] && [[ "$fileName" != "num_steps" ]]; then
                    continue 1
                  fi
                  ;;
                restore)
                  if [[ "$fileName" != *"_gf" ]] && [[ "$fileName" != "num_steps" ]]; then
                    continue 1
                  fi
                  ;;
              esac
            }
            check_file_type

            # Check if comparison failed
            check_fail() {
    					if [ "${PIPESTATUS[0]}" -ne 0 ]
    					then
    						echo "${scriptName}_${testtype}_${normtype}: FAIL" | tee -a $simulationLogFile

                # Skip to next test script
    						break 3
    					fi
            }

            # Check if a file exists on the user branch
            check_exists() {
              if [[ ! -f "$basetestfile" ]]; then
                echo "${fileName} exists on the baseline branch, but not on the user branch." | tee -a $simulationLogFile
                echo "${scriptName}_${testtype}_${normtype}: FAIL" | tee -a $simulationLogFile

                # Skip to next test script
                break 3
              fi
            }

  					# Compare last timestep of ROMSol
  					if [[ -d "$testFile" ]] && [[ "$fileName" == "ROMsol" ]]; then
  							echo "Comparing: "$fileName"/romS_$num_steps"
                basetestfile="$BASE_DIR/run/$fileName/romS_$num_steps"
                check_exists
  							$DIR/./fileComparator "$testFile/romS_$num_steps" "$basetestfile" "0.0" 2>&1 | tee -a $simulationLogFile
                check_fail

  			    # Compare FOM basis
  					elif [[ "$fileName" == "basis"* ]]; then
  						echo "Comparing: $fileName"
              basetestfile="$BASE_DIR/run/$fileName"
              check_exists

  						testFile=$(echo "$testFile" | cut -f 1 -d '.')
  						fileName=$(echo "$fileName" | cut -f 1 -d '.')
  						$DIR/./basisComparator "$testFile" "$BASE_DIR/run/$fileName" "1.0e-2" 2>&1 | tee -a $simulationLogFile
              check_fail

  					# Compare solutions, singular values, and number of time steps
  					elif [[ "$fileName" == "Sol"* ]] || [[ "$fileName" == "sVal"* ]] ||
  					[[ "$fileName" == "num_steps" ]] || [[ "$fileName" == *"_gf" ]]; then
  						echo "Comparing: $fileName"
              basetestfile="$BASE_DIR/run/$fileName"
              check_exists
  						$DIR/./fileComparator "$testFile" "$BASE_DIR/run/$fileName" "1.0e-7" 2>&1 | tee -a $simulationLogFile
              check_fail
  					fi
  				done

          # Passed
          echo "${scriptName}_${testtype}_${normtype}: PASS" | tee -a $simulationLogFile
        done
			done
		fi
	done
done
