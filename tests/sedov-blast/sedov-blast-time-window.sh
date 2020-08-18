NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -ef 0.9999 -nwin 4 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window.csv -writesol -romsvds
    ;;
  2)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -nwin 4 -twp twpTemp.csv -soldiff
    ;;
  3)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romhr -nwin 4 -twp twpTemp.csv -soldiff
    ;;
  4)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -restore -nwin 4 -twp twpTemp.csv
    ;;
esac
