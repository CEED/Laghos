NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -offline -writesol -romsvds -nwin 2 -tw "$BASE_DIR"/tests/taylor-green/taylor-green-time-window.csv -romsrhs
    ;;
  2)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romgs -soldiff -nwin 2 -twp twpTemp.csv -romsrhs
    ;;
  3)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romgs -romhrprep -nwin 2 -twp twpTemp.csv -romsrhs
    $LAGHOS_SERIAL -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romgs -romhr -nwin 2 -twp twpTemp.csv -romsrhs
    ;;
  4)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -restore -nwin 2 -twp twpTemp.csv -soldiff -romsrhs
    ;;
esac
