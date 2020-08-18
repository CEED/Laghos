# Regression Test Documentation

The script will git clone the master branch into tests/Laghos and run the regression tests on the user’s local machine, once with the user’s branch, and once with the master branch and compare the results. The csv and test scripts in the user’s branch are used for both simulations. If something fails along the way (i.e. a file is missing, the error bound is surpassed), the test fails with an appropriate error message and the user can look into the run directories to examine the files more closely. Since the tests are run on a user’s local workstation, both the baseline and the new user branch are run in the same environment. Currently, the regression tests will catch errors between branches, but will not catch errors relating to differing results on different machines.

The usage instructions are outputted whenever the script is run. Currently, there are include, exclude, and stopping options. Include is used if a user only wants to run some tests. Exclude is used if a user wants to run all tests except for some. The default is that all tests are run. The stop option is used to end the regression tests at the first test failure. This allows a user to look at the simulation data from both the user branch and the baseline branch, whereas without, it would be overwritten by the next test.

How to run the tests on both MAC and LC

1. If you have just changed branches, run "make clean".

2. Run "make" to make sure you your branch is up-to-date with any local changes.

3. ./tests/runRegressionTests.sh (if in the base directory) or ./runRegressionTests.sh (if in the tests directory). Look below for
test options.

4. Test commands/logs are stored in tests/results. Since each run overwrites the previous run, only the last run's data is saved
in run and tests/Laghos/run. Use option -f to stop at the first failure and look at the failed run's data.

5. To erase the regression test data, run from the base directory: make clean-regtest

How to add a non-time-windowing test

1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast.sh to your test directory, renaming it as desired.
3. Your tests will be run both serially and in parallel.
4. Choose a number for NUM_PARALLEL_PROCESSORS.
5. The tests will be run sequentially (1, 2, 3, 4, ...). Create as many tests as
desired following the given format.
6. All tests should have the format $HEADER laghos ...
7. Name your tests in testNames, which is an array of test names.

How to add a time-windowing test

1. Choose an appropriate sub-test directory.
2. Copy sedov-blast/sedov-blast-time-window.sh and sedov-blast/sedov-blast-time-window.csv
to your test directory, renaming them as desired.
3. Modify the csv to fit your problem.
5. For any offline commands, make sure to point to the csv with "$BASE_DIR"/tests/...
6. Follow the instructions for how to add a non-time-windowing test.

Here are some example runs and results:

./runRegressionTests.sh -> Run all tests.

./runRegressionTests.sh -f -> Run all tests, stopping at the first test failure.

./runRegressionTests.sh -i "sedov_blast gresho_vortices" -> Run sedov_blast and gresho_vortices.

./runRegressionTests.sh -e "taylor-green" -> Run all tests except taylor-green.

./runRegressionTests.sh -i "sedov_blast" -e "taylor-green" -> Error. -i and -e can not be used simultaneously.
