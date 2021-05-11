NUM_PARALLEL_PROCESSORS=8
testNames=(build_database use_database_online use_database_restore)
case $subTestNum in
  1)
    $GREEDY $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -build-database -writesol -romsvds -greedy-param-min 0.5 -greedy-param-max 2.5 -greedy-param-size 5 -greedysubsize 2 -greedyconvsize 3
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -online -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -restore -rdimx 2 -rdimv 12 -rdime 16 -soldiff
    ;;
esac
