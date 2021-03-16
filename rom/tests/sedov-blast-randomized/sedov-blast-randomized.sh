NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -romsvdrm -randdimx 2 -randdimv 12 -randdime 6 -romsrhs
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 11 -rdime 5 -sfacx 6 -sfacv 20 -sface 2 -soldiff -romsrhs
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romhrprep -rdimx 2 -rdimv 11 -rdime 5 -sfacx 6 -sfacv 20 -sface 2 -romsrhs
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romhr -rdimx 2 -rdimv 11 -rdime 5 -sfacx 6 -sfacv 20 -sface 2 -romsrhs
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 11 -rdime 5 -romhrprep -qdeim -romsrhs
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 11 -rdime 5 -romhr -qdeim -romsrhs
    ;;
  5)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 11 -rdime 5 -soldiff -romsrhs
    ;;
esac
