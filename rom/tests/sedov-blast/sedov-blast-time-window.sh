NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -ef 0.9999 -nwin 4 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window.csv -writesol -romsvds
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -nwin 4 -soldiff
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -romhrprep -nwin 4
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -romhr -nwin 4
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 4 -soldiff
    ;;
esac
