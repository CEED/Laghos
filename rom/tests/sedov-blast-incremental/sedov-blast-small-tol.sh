NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -no-romsns -romos -no-romsvds
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 7 -rdime 5 -soldiff -no-romsns -romos
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 7 -rdime 5 -rdimfv 9 -rdimfe 6 -romhrprep -nsamx 4 -nsamv 24 -nsame 32 -no-romsns -romos
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -romgs -rdimx 2 -rdimv 7 -rdime 5 -rdimfv 9 -rdimfe 6 -romhr -nsamx 4 -nsamv 24 -nsame 32 -no-romsns -romos
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 7 -rdime 5 -soldiff -no-romsns -romos
    ;;
esac
