NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -offline -writesol -romsns
    ;;
  2)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -online -rdimx 2 -rdimv 12 -rdime 6 -soldiff -romsns
    ;;
  3)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -online -rdimx 2 -rdimv 12 -rdime 6 -romhrprep -nsamx 4 -nsamv 24 -nsame 32 -romsns
    $LAGHOS_SERIAL -m data/rt2D.mesh -p 7 -tf 0.05 -online -rdimx 2 -rdimv 12 -rdime 6 -romhr -nsamx 4 -nsamv 24 -nsame 32 -romsns 
    ;;
  4)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -restore -rdimx 2 -rdimv 12 -rdime 6 -soldiff -romsns
    ;;
esac
