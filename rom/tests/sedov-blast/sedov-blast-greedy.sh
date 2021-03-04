NUM_PARALLEL_PROCESSORS=8
testNames=(build_database use_database_online use_database_restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -build-database -writesol -romsvds -greedysubsize 2 -greedyconvsize 3 -greedyfile "$BASE_DIR"/tests/sedov-blast/sedov-blast-greedy.csv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -online -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30 -greedyfile "$BASE_DIR"/tests/sedov-blast/sedov-blast-greedy.csv
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -restore -rdimx 2 -rdimv 12 -rdime 16 -soldiff -greedyfile "$BASE_DIR"/tests/sedov-blast/sedov-blast-greedy.csv
    ;;
esac
