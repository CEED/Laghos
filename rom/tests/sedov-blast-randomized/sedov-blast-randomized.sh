NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -romsvdrm -randdimx 2 -randdimv 12 -randdime 6
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 11 -rdime 5 -soldiff
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -romhrprep -rdimx 2 -rdimv 11 -rdime 5 -rdimfv 16 -rdimfe 9 -sfacx 6 -sfacv 20 -sface 2
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -romhr -rdimx 2 -rdimv 11 -rdime 5 -rdimfv 16 -rdimfe 9 -sfacx 6 -sfacv 20 -sface 2
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 11 -rdime 5 -rdimfv 16 -rdimfe 9 -romhrprep -qdeim
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 11 -rdime 5 -rdimfv 16 -rdimfe 9 -romhr -qdeim
    ;;
  5)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 11 -rdime 5 -soldiff
    ;;
esac
