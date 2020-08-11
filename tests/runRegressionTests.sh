#!/bin/bash

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

# Check whether tests exist
for simulation in "${testsToRun[@]}"
do

	# Check if simulation directory exists
	if [ ! -d $DIR/$simulation ]; then
		echo "$simulation test directory doesn't exist. Please try again."
		exit 1
	fi
done

###############################################################################
# COMPILATION
###############################################################################

echo "Setting up test suite"
echo "For detailed logs of the regression tests, please check tests/results."

# Create results directory
RESULTS_DIR=$DIR/results
if [ ! -d $RESULTS_DIR ]; then
  mkdir -p $RESULTS_DIR;
else
  rm -rf $RESULTS_DIR/*
fi

# Create regression test log file
setupLogFile=${RESULTS_DIR}/setup.log
touch $setupLogFile >> $setupLogFile 2>&1

# Save directory of the new Laghos executable
BASE_DIR=$DIR/..

# Save directory of the baseline Laghos executable
BASELINE_LAGHOS_DIR=$DIR/Laghos

# Get LIBS_DIR to run make depending on whether Gitlab is running
if [ -z "$CI_BUILDS_DIR" ]; then
	LIBS_DIR="$BASE_DIR/.."
else
	LIBS_DIR="$CI_BUILDS_DIR/$CI_PROJECT_NAME/env"
fi

# Compile the C++ comparators
echo $"Compiling the file and basis comparators" >> $setupLogFile 2>&1
g++ -std=c++11 -o $DIR/fileComparator $DIR/fileComparator.cpp >> $setupLogFile 2>&1
g++ -std=c++11 -o $DIR/basisComparator $DIR/basisComparator.cpp \
-I$LIBS_DIR/libROM -L$LIBS_DIR/libROM/build -lROM -Wl,-rpath,$LIBS_DIR/libROM/build >> $setupLogFile 2>&1

# Clone and compile rom-dev branch of Laghos
echo $"Cloning the baseline branch" >> $setupLogFile 2>&1
git -C $DIR clone -b rom-dev https://github.com/CEED/Laghos.git >> $setupLogFile 2>&1

# Copy user.mk
echo $"Copying user.mk to the baseline branch" >> $setupLogFile 2>&1
cp $DIR/../user.mk $DIR/Laghos/user.mk >> $setupLogFile 2>&1

# Check that rom-dev branch of Laghos is present
if [ ! -d $BASE_LAGHOS_DIR ]; then
	echo "Baseline Laghos directory could not be cloned" | tee -a $setupLogFile
	exit 1
fi

# Build the baseline Laghos executable
echo $"Building the baseline branch" >> $setupLogFile 2>&1
make --directory=$BASELINE_LAGHOS_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

# Check if make built correctly
if [ $? -ne 0 ]
then
	echo "The baseline branch failed to build" | tee -a $setupLogFile
  exit 1
fi

###############################################################################
# RUN TESTS
###############################################################################

case "$(uname -s)" in
    Linux*)
			SERIAL="srun -n 1 -p pdebug"
			PARALLEL="srun -n 8 -p pdebug";;
    Darwin*)
			SERIAL="mpirun -n 1"
			PARALLEL="mpirun -oversubscribe -n 8";;
    *)
			echo "The regression tests can only run on Linux and MAC."
			exit 1
esac


# Test number counter
testNum=0
testNumFail=0
testNumPass=0

# Test type
testtypes=(offline online romhr restore)

# Run all tests
for simulation in "${testsToRun[@]}"
do

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
        make --directory=$BASE_DIR LIBS_DIR="$LIBS_DIR" clean-exec >/dev/null 2>&1
        make --directory=$BASELINE_LAGHOS_DIR LIBS_DIR="$LIBS_DIR" clean-exec >/dev/null 2>&1

        # Run every test type
        for testtype in "${testtypes[@]}"
        do
					testNum=$((testNum+1))

					# Test failed boolean variable
					testFailed=false

					# Create simulation results log file
					simulationLogFile="${RESULTS_DIR}/${scriptName}-${testtype}-${normtype}.log"
					touch $simulationLogFile

					set_pass() {
						testNumPass=$((testNumPass+1))
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testtype}-${normtype}: PASS"
						echo "${scriptName}-${testtype}-${normtype}: PASS" >> $simulationLogFile
					}

					set_fail() {
						testFailed=true
						testNumFail=$((testNumFail+1))
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testtype}-${normtype}: FAIL"
						echo "${scriptName}-${testtype}-${normtype}: FAIL" >> $simulationLogFile
					}

					echo -n "$testNum. ${scriptName}-${testtype}-${normtype}: RUNNING"

  				# Run simulation from rom-dev branch
  			  echo $"Running baseline simulation for comparison" >> $simulationLogFile 2>&1
  				(cd $BASELINE_LAGHOS_DIR && . "$script") >> $simulationLogFile 2>&1

					# Check if simulation failed
					if [ "$?" -ne 0 ]
					then
						echo "Something went wrong running the baseline simulation with the
            test script: $scriptName" >> $simulationLogFile 2>&1
            set_fail

            # Skip to next test
						continue 1
					fi

  				# # Run simulation from current branch
  				echo $"Running new simulation for regression testing" >> $simulationLogFile 2>&1
  				(cd $BASE_DIR && . "$script") >> $simulationLogFile 2>&1

					# Check if simulation failed
          if [ "$?" -ne 0 ]
					then
						echo "Something went wrong running the new user branch simulation with the
             test script: $scriptName" >> $simulationLogFile 2>&1
            set_fail

            # Skip to next test
						continue 1
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
									if [[ "$scriptName" == *"parallel"* ]]; then
										if [[ "$fileName" != "num_steps" ]]; then
	                    continue 1
	                  fi
									else
										if [[ "$fileName" != *"_gf" ]] && [[ "$fileName" != "num_steps" ]]; then
	                    continue 1
	                  fi
									fi
                  ;;
              esac
            }
            check_file_type

            # Check if comparison failed
            check_fail() {
    					if [ "${PIPESTATUS[0]}" -ne 0 ]
    					then
    						set_fail

                # Skip to next test
								break 1
    					fi
            }

            # Check if a file exists on the user branch
            check_exists() {
              if [[ ! -f "$basetestfile" ]]; then
                echo "${fileName} exists on the baseline branch, but not on the user branch." >> $simulationLogFile 2>&1
                set_fail

                # Skip to next test
								break 1
              fi
            }

  					# Compare last timestep of ROMSol
  					if [[ -d "$testFile" ]] && [[ "$fileName" == "ROMsol" ]]; then
  							echo "Comparing: "$fileName"/romS_$num_steps" >> $simulationLogFile 2>&1
                basetestfile="$BASE_DIR/run/$fileName/romS_$num_steps"
                check_exists
								if [[ "$scriptName" == *"parallel"* ]]; then
  								$($DIR/./fileComparator "$testFile/romS_$num_steps" "$basetestfile" "3.0" >> $simulationLogFile 2>&1)
								else
  								$($DIR/./fileComparator "$testFile/romS_$num_steps" "$basetestfile" "1.0e-7" >> $simulationLogFile 2>&1)
								fi
                check_fail

  			    # Compare FOM basis
  					elif [[ "$fileName" == "basis"* ]]; then
  						echo "Comparing: $fileName" >> $simulationLogFile 2>&1
              basetestfile="$BASE_DIR/run/$fileName"
              check_exists
							testFile="${testFile%.*}"
							basetestfile="${basetestfile%.*}"
  						$($DIR/./basisComparator "$testFile" "$basetestfile" "1.0e-2" >> $simulationLogFile 2>&1)
              check_fail

  					# Compare solutions, singular values, and number of time steps
  					elif [[ "$fileName" == "Sol"* ]] || [[ "$fileName" == "sVal"* ]] ||
  					[[ "$fileName" == "num_steps" ]] || [[ "$fileName" == *"_gf" ]] ; then
  						echo "Comparing: $fileName" >> $simulationLogFile 2>&1
              basetestfile="$BASE_DIR/run/$fileName"
              check_exists
							if [[ "$scriptName" == *"parallel"* ]]; then
								$($DIR/./fileComparator "$testFile" "$basetestfile" "3.0" >> $simulationLogFile 2>&1)
							else
								$($DIR/./fileComparator "$testFile" "$basetestfile" "1.0e-7" >> $simulationLogFile 2>&1)
							fi
              check_fail
  					fi
  				done

          # Passed
					if [ "$testFailed" == false ]; then
						set_pass
					fi
        done
			done
		fi
	done
done
echo "${testNumPass} passed, ${testNumFail} failed out of ${testNum} tests"
if [ $testNumFail -ne 0 ]; then
	exit 1
fi
