normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    $PARALLEL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -offline -writesol -romsvds -normtype "$normtype"
    ;;
  online)
    $PARALLEL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -rdimx 1 -rdimv 4 -rdime 3 -soldiff -normtype "$normtype"
    ;;
  romhr)
    $PARALLEL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -rdimx 1 -rdimv 4 -rdime 3 -romhr -nsamx 6 -nsamv 448 -nsame 10 -soldiff -normtype "$normtype"
    ;;
  restore)
    $PARALLEL laghos -p 3 -m "$BASE_DIR"/data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -restore -rdimx 1 -rdimv 4 -rdime 3 -normtype "$normtype"
    ;;
esac
