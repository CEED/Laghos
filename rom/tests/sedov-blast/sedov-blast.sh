NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -romsvds -romsrhs
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30 -soldiff -romsrhs
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 16 -romhrprep -nsamx 4 -nsamv 24 -nsame 32 -romsrhs
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 16 -romhr -nsamx 4 -nsamv 24 -nsame 32 -romsrhs
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 16 -romhrprep -qdeim -romsrhs
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 12 -rdime 16 -romhr -qdeim -romsrhs
    ;;
  5)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 12 -rdime 16 -soldiff -romsrhs
    ;;
esac
