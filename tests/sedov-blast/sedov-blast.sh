normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    $SERIAL laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -normtype "$normtype" -writesol -romsvds
    ;;
  online)
    $SERIAL laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30 -normtype "$normtype" -soldiff
    ;;
  romhr)
    $SERIAL laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -romhr -nsamx 4 -nsamv 24 -nsame 32 -normtype "$normtype" -soldiff
    ;;
  restore)
    $SERIAL laghos -m "$BASE_DIR"/data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 12 -rdime 16 -normtype "$normtype"
    ;;
esac
