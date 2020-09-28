# Regression Test Documentation

The script will git clone the master branch into tests/Laghos and run the regression tests on the user’s local machine, once with the user’s branch, and once with the master branch and compare the results. The csv and test scripts in the user’s branch are used for both simulations. If something fails along the way (i.e. a file is missing, the error bound is surpassed), the test fails with an appropriate error message and the user can look into the run directories to examine the files more closely. Since the tests are run on a user’s local workstation, both the baseline and the new user branch are run in the same environment. Currently, the regression tests will catch errors between branches, but will not catch errors relating to differing results on different machines.

The usage instructions are outputted whenever the script is run. Currently, there are include, exclude, and stopping options. Include is used if a user only wants to run some tests. Exclude is used if a user wants to run all tests except for some. The default is that all tests are run. The stop option is used to end the regression tests at the first test failure. This allows a user to look at the simulation data from both the user branch and the baseline branch, whereas without, it would be overwritten by the next test.

How to run the tests on LC

1. Make sure your user branch is up-to-date with any recent rom-dev commits and do a git merge if necessary. Otherwise, your tests will most likely fail.
2. sbatch tests/runRegressionTests.sh (if in the base directory) or sbatch runRegressionTests.sh (if in the tests directory). make clean-regtest, make clean, make, and make merge will be run automatically. The baseline branch will be git cloned and rebuilt each time the script is called. The different tests will automatically run in parallel (can not be turned off). Look below for test options.
3. The slurm output file will be stored in sbatch.log in the directory you ran the previous command from.
4. Test commands/logs are stored in tests/results. Each run's data is in it's own directory in run and tests/Laghos/run.
5. To find the particular error and where it occurred, scroll to the bottom of each test file. You can then look through the data files in Laghos/run and Laghos/run/tests/run to see what went wrong with each individual test and to compare the numerical values yourself. Use -f to avoid online-non-romhr failing commands being overwritten by the subsequent romhr test command.
6. To erase the regression test data, run from the base directory: make clean-regtest

How to run the tests on MAC

1. Follow step 1 above.
2. ./tests/runRegressionTests.sh (if in the base directory) or ./runRegressionTests.sh (if in the tests directory). Look below for test options.
3. Follow steps 4-6 from the instructions above.

How to add a non-time-windowing test

1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast.sh to your test directory, renaming it as desired.
Use it as an example.
3. Choose a number for NUM_PARALLEL_PROCESSORS. Your tests will be run both serially and in parallel. If you remove this line, your tests will only run in serial.
4. The tests will be run sequentially (1, 2, 3, 4, ...). Create as many tests as
desired following the given format.
5. Any laghos commands should have the format $LAGHOS ...
6. Name your tests in testNames, which is an array of test names. If your test name
is multiple words, separate the words with an underscore, rather than a space.

How to add a time-windowing test

1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast-time-window.sh and sedov-blast/sedov-blast-time-window.csv
to your test directory, renaming them as desired. Use it as an example.
3. Modify the csv to fit your problem.
4. For any offline commands, make sure to point to the csv with "$BASE_DIR"/tests/...
5. Follow the instructions for how to add a non-time-windowing test.

How to add a non-time-windowing parameter variation test

1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast-parameter-variation.sh to your test directory, renaming
the file as desired. Use it as an example.
3. Since what we want to compare is the FOM solution and ROM solution, the tests
only need to be split into two sub-tests, like sedov-blast-parameter-variation.sh.
4. Follow the instructions for how to add a non-time-windowing test.

How to add a time-windowing parameter variation test
1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast-parameter-variation-time-window.sh
3. Follow the instructions of both the time-windowing test and non-time-windowing
parameter variation test to create your test script.

Here are some example runs and results:

./runRegressionTests.sh -> Run all tests.

./runRegressionTests.sh -f -> Run all tests, stopping at the first test failure on each processor.

./runRegressionTests.sh -t -> Run all tests, but uses the current user branch (as on the Github repo) as the baseline to verify new tests or uncommitted local changes.

./runRegressionTests.sh -d -> Run all tests, but do not do comparisons. Failures only occur if the simulations do not run successfully.

./runRegressionTests.sh -i "sedov_blast gresho_vortices" -> Run sedov_blast and gresho_vortices.

./runRegressionTests.sh -e "taylor-green" -> Run all tests except taylor-green.

./runRegressionTests.sh -i "sedov_blast" -e "taylor-green" -> Error. -i and -e can not be used simultaneously.
