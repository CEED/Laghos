#!/bin/bash

#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -p pdebug
#SBATCH -o sbatch.log
#SBATCH --open-mode truncate

###############################################################################
# SET UP
###############################################################################
usage() {
	echo "Unknown option. Refer to REGRESSIONTEST.md"
	exit 1
}

# Stop at failure
stopAtFailure=false
dryRun=false
useUserAsBaseline=false

# Skip setup (git pull, make))
${skipSetup:=false}
# Get options
while getopts ":i:e:th:dh:fh" o;
do
	case "${o}" in
		i)
			i=${OPTARG}
      ;;
    e)
      e=${OPTARG}
      ;;
		t)
      useUserAsBaseline=true
      ;;
		d)
			dryRun=true
			;;
		f)
			stopAtFailure=true
			;;
    *)
      usage
      ;;
    esac
done
shift $((OPTIND-1))

# If both include and exclude are set, fail
if [[ -n "${i}" ]] && [[ -n "${e}" ]]; then
    usage
		exit 1
fi

# Save directory of this script
if [[ -f "$PWD/runRegressionTests.sh" ]]; then
	DIR=$PWD
elif [[ -f "$PWD/tests/runRegressionTests.sh" ]]; then
	DIR=$PWD/tests
else
	echo "Run tests from the Laghos or Laghos/tests directory"
fi

# Accumulate tests to run
testsToRun=( $i )
testsToExclude=( $e )
if [[ ${#testsToRun[@]} -eq 0 ]];
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

# Create results directory
RESULTS_DIR=$DIR/results

# Save directory of the new Laghos executable
BASE_DIR=$DIR/..

# Save directory of the baseline Laghos executable
BASELINE_LAGHOS_DIR=$DIR/Laghos

# Get LIBS_DIR to run make depending on whether Gitlab is running
if [[ -z "$CI_BUILDS_DIR" ]]; then
	LIBS_DIR="$BASE_DIR/.."
else
	LIBS_DIR="$CI_BUILDS_DIR/$CI_PROJECT_NAME/env"
fi

# If skipping setup, don't do make
if [[ "$skipSetup" == "false" ]];
then

	# Cleaning the regression test directory
	make clean --directory=$BASE_DIR LIBS_DIR="$LIBS_DIR" > /dev/null

	echo "Setting up test suite"
	echo "For detailed logs of the regression tests, please check tests/results."

	if [ ! -d $RESULTS_DIR ]; then
	  mkdir -p $RESULTS_DIR;
	else
	  rm -rf $RESULTS_DIR/*
	fi

	# Create regression test log file
	setupLogFile=${RESULTS_DIR}/setup.log
	touch $setupLogFile >> $setupLogFile 2>&1

	echo "Cleaned the regression test directory and user branch build" >> $setupLogFile 2>&1

	# Compile the C++ comparators
	echo "Compiling the file, basis, and solution comparators" >> $setupLogFile 2>&1
	make regtest --directory=$BASE_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

	# Check if make built correctly
	if [[ $? -ne 0 ]];
	then
		echo "The regtest comparators failed to build." | tee -a $setupLogFile
		exit 1
	fi

	# If skipping setup, don't do make
	if [[ "$useUserAsBaseline" == "true" ]];
	then

		# Clone and compile user branch as baseline
		echo "Cloning $(git branch | sed -n '/\* /s///p') as the baseline branch" >> $setupLogFile 2>&1
		git -C $DIR clone -b "$(git branch | sed -n '/\* /s///p')" https://github.com/CEED/Laghos.git >> $setupLogFile 2>&1
	else

		# Clone and compile rom-dev branch of Laghos as baseline
		echo "Cloning rom-dev as the baseline branch" >> $setupLogFile 2>&1
		git -C $DIR clone -b rom-dev https://github.com/CEED/Laghos.git >> $setupLogFile 2>&1
	fi

	# Check that rom-dev branch of Laghos is present
	if [ ! -d $BASE_LAGHOS_DIR ]; then
		echo "Baseline Laghos directory could not be cloned" | tee -a $setupLogFile
		exit 1
	fi

	# Copy user.mk
	echo "Copying user.mk to the baseline branch" >> $setupLogFile 2>&1
	cp $DIR/../user.mk $DIR/Laghos/user.mk >> $setupLogFile 2>&1

	# Build the user branch executable
	echo "Building the user branch" >> $setupLogFile 2>&1
	make --directory=$BASE_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

	# Check if make built correctly
	if [[ $? -ne 0 ]];
	then
		echo "The user branch failed to build." | tee -a $setupLogFile
		exit 1
	fi

	# Build merge
	make merge --directory=$BASE_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

	# Check if make built correctly
	if [[ $? -ne 0 ]];
	then
		echo "The user branch merge executable failed to build." | tee -a $setupLogFile
		exit 1
	fi

	# Build the baseline Laghos executable
	echo "Building the baseline branch" >> $setupLogFile 2>&1
	make --directory=$BASELINE_LAGHOS_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

	# Check if make built correctly
	if [[ $? -ne 0 ]];
	then
		echo "The baseline branch failed to build." | tee -a $setupLogFile
	  exit 1
	fi

	# Build merge
	make merge --directory=$BASELINE_LAGHOS_DIR LIBS_DIR="$LIBS_DIR" >> $setupLogFile 2>&1

	# Check if make built correctly
	if [[ $? -ne 0 ]];
	then
		echo "The baseline branch merge executable failed to build." | tee -a $setupLogFile
		exit 1
	fi
fi

# If running on slurm, parallelize the tests.
if [[ -z "$SLURM" ]]; then
	if [[ $0 == *"slurm"* ]]; then
		SLURM=true
		for simulation in "${testsToRun[@]}"
		do
			echo "Forking child. Check ${RESULTS_DIR}/${simulation}-results.log for immediate results."
			if [[ "$stopAtFailure" == "true" ]];
			then
				if [[ "$dryRun" == "true" ]];
				then
					skipSetup=true SLURM=true $DIR/runRegressionTests.sh -f -d -i $simulation >> ${RESULTS_DIR}/${simulation}-results.log 2>&1 &
				else
					skipSetup=true SLURM=true $DIR/runRegressionTests.sh -f -i $simulation >> ${RESULTS_DIR}/${simulation}-results.log 2>&1 &
				fi
			else
				if [[ "$dryRun" == "true" ]];
				then
					skipSetup=true SLURM=true $DIR/runRegressionTests.sh -d -i $simulation >> ${RESULTS_DIR}/${simulation}-results.log 2>&1 &
				else
					skipSetup=true SLURM=true $DIR/runRegressionTests.sh -i $simulation >> ${RESULTS_DIR}/${simulation}-results.log 2>&1 &
				fi
			fi
		done
		echo "After all processes are finished, results will be concatenated and outputted to ${RESULTS_DIR}/sbatch-results.log."
		wait
		echo "Finished. Check ${RESULTS_DIR}/sbatch-results.log"
		cat ${RESULTS_DIR}/*-results.log > ${RESULTS_DIR}/sbatch-results.log
		exit 0
	else
		SLURM=false
	fi
fi

###############################################################################
# RUN TESTS
###############################################################################

# Check machine
case "$(uname -s)" in
    Linux*)
		  COMMAND="srun -p pdebug";;
    Darwin*)
		  COMMAND="mpirun -oversubscribe";;
    *)
			echo "The regression tests can only run on Linux and MAC."
			exit 1
esac

# Test number counter
testNum=0
testNumFail=0
testNumPass=0

# Run all tests
for simulation in "${testsToRun[@]}"
do
	echo "$simulation"

	# Run every script in each test directory
	for script in ${DIR}/${simulation}/*;
	do

		if [[ "$script" == *".sh" ]]; then

      # Get script name without extension
			scriptName=$(basename $script)
      scriptName="${scriptName%.*}"

			subTestNum=0
			parallel=false
			NUM_PARALLEL_PROCESSORS=0

			# Get test names
			. "$script"

			while true;
			do

				if [[ "$parallel" == "false" ]];
				then
					HEADER="$COMMAND -n 1"
					testName="${testNames[$subTestNum]}"
					OUTPUT_DIR=${scriptName}
				else
					HEADER="$COMMAND -n $NUM_PARALLEL_PROCESSORS"
					testName="${testNames[$subTestNum]}-parallel"
					OUTPUT_DIR="${scriptName}-parallel"
				fi
				MERGE="$HEADER ./merge -o ${OUTPUT_DIR}"
				LAGHOS="$HEADER laghos -o ${OUTPUT_DIR}"

				# Update subtest numbers
				subTestNum=$((subTestNum+1))

				# Get testtype
				RAN_COMMAND=$(awk "/$subTestNum\)/{f=1;next} /;;/{f=0} f" $script | grep -F '$LAGHOS')
				if [[ $RAN_COMMAND == *"writesol"* ]]; then
					testtype=fom
				elif [[ $RAN_COMMAND == *"online"* ]]; then
					testtype=online
					rm -rf ${BASELINE_LAGHOS_DIR}/run/${OUTPUT_DIR}/ROMsol/*
					rm -rf ${BASE_DIR}/run/${OUTPUT_DIR}/ROMsol/*
				elif [[ $RAN_COMMAND == *"restore"* ]]; then
					testtype=restore
				else
					if [[ "$parallel" == "false" ]] && [[ $NUM_PARALLEL_PROCESSORS -ne 0 ]];
					then
						parallel=true
						subTestNum=0
						continue
					else
						break
					fi
				fi

				# Update test numbers
				testNum=$((testNum+1))

				# Test failed boolean variable
				testFailed=false

				# Create simulation results log file
				simulationLogFile="${RESULTS_DIR}/${scriptName}-${testName}.log"
				touch $simulationLogFile

				set_pass() {
					testNumPass=$((testNumPass+1))
					if [[ $SLURM == "true" ]]; then
						echo "$testNum. ${scriptName}-${testName}: PASS"
					else
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testName}: PASS"
					fi
					echo "${scriptName}-${testName}: PASS" >> $simulationLogFile
				}

				set_fail() {
					testFailed=true
					testNumFail=$((testNumFail+1))
					if [[ $SLURM == "true" ]]; then
						echo "$testNum. ${scriptName}-${testName}: FAIL"
					else
						echo -e "\\r\033[0K$testNum. ${scriptName}-${testName}: FAIL"
					fi
					echo "${scriptName}-${testName}: FAIL" >> $simulationLogFile
					if [[ "$stopAtFailure" == "true" ]];
					then
						exit 1
					fi
				}
				if [[ $SLURM == "false" ]]; then
					echo -n "$testNum. ${scriptName}-${testName}: RUNNING"
				fi

				# Run simulation from rom-dev branch
				echo "Running baseline simulation for comparison" >> $simulationLogFile 2>&1
				(cd $BASELINE_LAGHOS_DIR && set -o xtrace && . "$script") >> $simulationLogFile 2>&1

				# Check if simulation failed
				if [[ "$?" -ne 0 ]];
				then
					echo "Something went wrong running the baseline simulation with the
					test script: $scriptName. Try running 'make clean' and 'make'." >> $simulationLogFile 2>&1
					set_fail

					# Skip to next test
					continue 1
				fi

				# # Run simulation from current branch
				echo "Running new simulation for regression testing" >> $simulationLogFile 2>&1
				(cd $BASE_DIR && set -o xtrace && . "$script") >> $simulationLogFile 2>&1

				# Check if simulation failed
				if [[ "$?" -ne 0 ]];
				then
					echo "Something went wrong running the new user branch simulation with the
					 test script: $scriptName" >> $simulationLogFile 2>&1
					set_fail

					# Skip to next test
					continue 1
				fi

				# If doing dry run, skip comparisons
				if [[ "$dryRun" == "false" ]]; then

					# Necessary for now, mkdir in C++ does something weird where this bash can't find the file
					ls $BASELINE_LAGHOS_DIR/run > /dev/null
					ls $BASE_DIR/run > /dev/null

					# Find number of steps simulation took in rom-dev to compare final timestep later
					cmp -s "$BASELINE_LAGHOS_DIR/run/${OUTPUT_DIR}/num_steps" "$BASE_DIR/run/${OUTPUT_DIR}/num_steps" > /dev/null
					if [[ $? -eq 1 ]]; then
						echo "The number of time steps are different from the baseline." >> $simulationLogFile 2>&1
						set_fail
						continue 1
					fi
					num_steps="$(cat $BASELINE_LAGHOS_DIR/run/${OUTPUT_DIR}/num_steps)"

					# After simulations complete, compare results
					for baselineTestFile in $BASELINE_LAGHOS_DIR/run/${OUTPUT_DIR}/*
					do
						fileName="$(basename "$baselineTestFile")"

						# Skip if not correct file type to compare
						check_file_type() {
							if [[ $testtype == "fom" ]]; then
								if [[ "$fileName" != "basis"*".000000" ]] && [[ "$fileName" != "Sol"*".000000" ]] &&
								[[ "$fileName" != "sVal"*".000000" ]]; then
									continue 1
								fi
							elif [[ $testtype == "online" ]]; then
								if [[ "$fileName" != *"norms.000000" ]] && [[ "$fileName" != "ROMsol" ]]; then
									continue 1
								fi
							elif [[ $testtype == "restore" ]]; then
								if [[ "$fileName" != *"_gf.000000" ]]; then
									continue 1
								fi
							fi
						}
						check_file_type

						# Check if comparison failed
						check_fail() {
							if [[ "${PIPESTATUS[0]}" -ne 0 ]];
							then
								set_fail

								# Skip to next test
								break 1
							fi
						}

						# Check if a file exists on the user branch
						check_exists() {
							if [ ! -f $targetTestFile ]; then
								echo "${fileName} exists on the baseline branch, but not on the user branch." >> $simulationLogFile 2>&1
								set_fail

								# Skip to next test
								break 1
							fi
						}

						# Compare last timestep of ROMSol solutions
						if [[ -d "$baselineTestFile" ]] && [[ "$fileName" == "ROMsol" ]]; then
								echo "Comparing: "$fileName"/romS_$num_steps" >> $simulationLogFile 2>&1
								targetTestFile="$BASE_DIR/run/${OUTPUT_DIR}/$fileName/romS_$num_steps"
								check_exists
								baselineTestFile="$baselineTestFile/romS_$num_steps"
								if [[ "$parallel" == "true" ]]; then
									$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-5" "1" >> $simulationLogFile 2>&1)
								else
									$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-5" "1" >> $simulationLogFile 2>&1)
								fi
								check_fail

						# Compare FOM basis
						elif [[ "$fileName" == "basis"* ]]; then
							echo "Comparing: ${fileName%.*}" >> $simulationLogFile 2>&1
							targetTestFile="$BASE_DIR/run/${OUTPUT_DIR}/$fileName"
							check_exists
							baselineTestFile="${baselineTestFile%.*}"
							targetTestFile="${targetTestFile%.*}"
							if [[ "$parallel" == "true" ]]; then
								$($HEADER $DIR/./basisComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" "$NUM_PARALLEL_PROCESSORS" >> $simulationLogFile 2>&1)
							else
								$($DIR/./basisComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" "1" >> $simulationLogFile 2>&1)
							fi
							check_fail

						# Compare singular values and norms
					elif [[ "$fileName" == "sVal"* ]] ||	[[ "$fileName" == *"norms"* ]]; then
							echo "Comparing: ${fileName%.*}" >> $simulationLogFile 2>&1
							targetTestFile="$BASE_DIR/run/${OUTPUT_DIR}/$fileName"
							check_exists
							if [[ "$parallel" == "true" ]]; then
								$($DIR/./fileComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" >> $simulationLogFile 2>&1)
							else
								$($DIR/./fileComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" >> $simulationLogFile 2>&1)
							fi
							check_fail

						# Compare solutions
						elif [[ "$fileName" == "Sol"* ]] ; then
							echo "Comparing: ${fileName%.*}" >> $simulationLogFile 2>&1
							targetTestFile="$BASE_DIR/run/${OUTPUT_DIR}/$fileName"
							check_exists
							if [[ "$parallel" == "true" ]]; then
								$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" "$NUM_PARALLEL_PROCESSORS" >> $simulationLogFile 2>&1)
							else
								$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-7" "1" >> $simulationLogFile 2>&1)
							fi
							check_fail

						# Compare _gf
					elif [[ "$fileName" == *"_gf"* ]] ; then
							echo "Comparing: $fileName" >> $simulationLogFile 2>&1
							targetTestFile="$BASE_DIR/run/${OUTPUT_DIR}/$fileName"
							check_exists
							if [[ "$parallel" == "true" ]]; then
								$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-5" "$NUM_PARALLEL_PROCESSORS" >> $simulationLogFile 2>&1)
							else
								$($DIR/./solutionComparator "$baselineTestFile" "$targetTestFile" "1.0e-5" "1" >> $simulationLogFile 2>&1)
							fi
							check_fail
						fi
					done

				fi

				# Passed
				if [[ "$testFailed" == false ]]; then
					set_pass
				fi
			done
		fi
	done
done


echo "${testNumPass} passed, ${testNumFail} failed out of ${testNum} tests"
if [[ $testNumFail -ne 0 ]]; then
	exit 1
fi
