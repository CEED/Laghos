normtype=${normtype:-"l2"}
case $testtype in
  offline)
    ./laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -offline -writesol -romsvds -nwin 2 -tw "$BASE_DIR"/tests/taylor-green/tw2taylor-green.csv -normtype "$normtype"
    ;;
  online)
    ./laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -soldiff -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  romhr)
    ./laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -online -romhr -soldiff -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  restore)
    ./laghos -p 0 -rs 2 -iv -cfl 0.5 -tf 0.008 -pa -restore -nwin 2 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
esac
