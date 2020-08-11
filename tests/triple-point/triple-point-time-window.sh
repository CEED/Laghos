normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    $SERIAL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -offline -writesol -romsvds -ef 0.9999 -nwin 4 -tw "$BASE_DIR"/tests/triple-point/triple-point-time-window.csv -normtype "$normtype"
    ;;
  online)
    $SERIAL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -soldiff -nwin 4 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  romhr)
    $SERIAL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -soldiff -nwin 4 -romhr -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
  restore)
    $SERIAL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -restore -nwin 4 -twp "$BASE_DIR"/twpTemp.csv -normtype "$normtype"
    ;;
esac
