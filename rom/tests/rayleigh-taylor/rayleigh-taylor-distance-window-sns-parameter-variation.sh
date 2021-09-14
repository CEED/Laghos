NUM_PARALLEL_PROCESSORS=8
testNames=(fom romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -offline -romsns -rpar 0 -loctype distance
    $MERGE -nset 1 -romsns -nwinsamp 10 -loctype distance
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -af 0.33 -writesol
    ;;
  2)
    $LAGHOS -m data/rt2D.mesh -p 7 -tf 0.05 -online -sfacx 10 -sfacv 10 -sface 10 -nwin 2 -romsns -romhrprep -af 0.33 -loctype distance
    $LAGHOS_SERIAL -m data/rt2D.mesh -p 7 -tf 0.05 -online -sfacx 10 -sfacv 10 -sface 10 -nwin 2 -romsns -romhr -af 0.33 -loctype distance
    ;;
  3)
    $LAGHOS -m data/rt2D.mesh -p 7 -restore -nwin 2 -soldiff -romsns -loctype distance
    ;;
esac
