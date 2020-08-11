normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    $PARALLEL laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -offline -writesol -romsvds -nwin 2 -tw "$BASE_DIR"/tests/taylor-green/taylor-green-time-window.csv -normtype "$normtype"
    ;;
  online)
    $PARALLEL laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -soldiff -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  romhr)
    $PARALLEL laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romhr -soldiff -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  restore)
    $PARALLEL laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -restore -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
esac
