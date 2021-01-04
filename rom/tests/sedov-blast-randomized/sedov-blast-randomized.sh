NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr qdeim restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -writesol -romsvdrm
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -nsamx 12 -nsamv 184 -nsame 30 -soldiff
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -romhr -nsamx 4 -nsamv 24 -nsame 32 -soldiff
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 2 -rdimv 12 -rdime 16 -romhr -qdeim -soldiff
    ;;
  5)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -restore -rdimx 2 -rdimv 12 -rdime 16
    ;;
esac
