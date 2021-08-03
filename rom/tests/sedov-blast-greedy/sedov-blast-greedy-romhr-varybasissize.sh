NUM_PARALLEL_PROCESSORS=8
testNames=(build_database use_database_online use_database_restore)
case $subTestNum in
  1)
    . $GREEDY -m data/cube01_hex.mesh -pt 211 -tf 0.01 -build-database -writesol -romsvds -greedy-param-min 0.5 -greedy-param-max 2.5 -greedy-param-size 5 -greedysubsize 2 -greedyconvsize 3 -greedyerrindtype varyBasisSize -romhr
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -online -rdimx 7 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 15 -nsamx 12 -nsamv 184 -nsame 30 -romhrprep
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -online -rdimx 7 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 15 -nsamx 12 -nsamv 184 -nsame 30 -romhr
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -use-database -bef 0.9 -restore -rdimx 7 -rdimv 12 -rdime 6 -rdimfv 16 -rdimfe 15
    ;;
esac
