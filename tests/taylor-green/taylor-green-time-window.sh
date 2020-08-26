NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $HEADER laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -offline -writesol -romsvds -nwin 2 -tw "$BASE_DIR"/tests/taylor-green/taylor-green-time-window.csv
    ;;
  2)
    $HEADER laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -soldiff -nwin 2 -twp twpTemp.csv
    ;;
  3)
    $HEADER laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romhr -soldiff -nwin 2 -twp twpTemp.csv
    ;;
  4)
    $HEADER laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -restore -nwin 2 -twp twpTemp.csv
    ;;
esac
