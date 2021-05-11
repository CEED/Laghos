NUM_PARALLEL_PROCESSORS=4
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -offline -ef 0.9999 -writesol -nwinsamp 10 -romsns
    ;;
  2)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -online -nwin 2 -soldiff -romsns
    ;;
  3)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -online -sfacv 10 -sface 10 -nwin 2 -romsns -romhrprep 
    $LAGHOS_SERIAL -m data/rt2D.mesh -p 7 -tf 0.05 -online -sfacv 10 -sface 10 -nwin 2 -romsns -romhr 
    ;;
  4)
    $LAGHOS -m data/rt2D.mesh -p 7 -restore -nwin 2 -soldiff -romsns 
    ;;
esac
