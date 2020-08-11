normtype=${normtype:-"l2"}
set -o xtrace
case $testtype in
  offline)
    $PARALLEL laghos -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -offline -writesol -romsvds -normtype "$normtype"
    ;;
  online)
    $PARALLEL laghos -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -soldiff -normtype "$normtype"
    ;;
  romhr)
    $PARALLEL laghos -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -online -rdimx 2 -rdimv 6 -rdime 2 -soldiff -romhr -nsamx 96 -nsamv 320 -nsame 64 -normtype "$normtype"
    ;;
  restore)
    $PARALLEL laghos -p 0 -rs 1 -iv -cfl 0.5 -tf 0.07 -pa -restore -rdimx 2 -rdimv 6 -rdime 2 -normtype "$normtype"
    ;;
esac
