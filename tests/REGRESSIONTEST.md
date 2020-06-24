# Regression Test Documentation

The script will git clone the master branch into tests/Laghos and run the regression tests on the user’s local machine, once with the user’s branch, and once with the master branch and compare the results. The csv and test scripts in the user’s branch are used for both simulations. If something fails along the way (i.e. a file is missing, the error bound is surpassed), the test fails with an appropriate error message and the user can look into the run directories to examine the files more closely. Since the tests are run on a user’s local workstation, both the baseline and the new user branch are run in the same environment. Currently, the regression tests will catch errors between branches, but will not catch errors relating to differing results on different machines.

The usage instructions are outputted whenever the script is run. Currently, there are include, exclude, and normtype options. Include is used if a user only wants to run some tests. Exclude is used if a user wants to run all tests except for some. The default is that all tests are run. Without a normtype option, the tests default to l2, but a user can use any normtype or optionally add multiple to run the tests with both l2 and max sequentially, for example.

Here are some example runs and results:

./runRegressionTests.sh -> Run all tests with l2 norm.

./runRegressionTests.sh -i "sedov_blast gresho_vortices" -> Run sedov_blast and gresho_vortices with l2 norm.

./runRegressionTests.sh -e "taylor-green" -> Run all tests with l2 norm except taylor-green.

./runRegressionTests.sh -i "sedov_blast" -e "taylor-green" -> Error. -i and -e can not be used simultaneously.

./runRegressionTests.sh -i "triple-point" -n "l1 max" -> Run triple-point with l1 norm, then max norm.

1. To add a regression test to the test suite, create a test directory within the tests directory. Within that directory, include a script with your two commands (offline & online) along with any .csv time-windowing file that is needed. Two minor changes have to be made to these commands within the script. The first is initializing a normtype variable and the second is initializing all paths with $BASE_DIR. These changes allow the regression test to be run with any normtype and will ensure that only the script, data, and csv files in the new branch will be used. sedov_blast can be used as an example.

2. Multiple scripts can be included in each directory, but it is important that each script only contain a single offline and online command.

3. To run the regression tests, run the following command: ./tests/runRegressionTests.sh (if in the base directory) or ./runRegressionTests.sh (if in the tests directory)

4. To erase the regression test data, run from the base directory: make clean-regtest
