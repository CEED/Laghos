NUM_PARALLEL_PROCESSORS=1
testNames=(offline online restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.02 -offline -romsns -rpar 0 -sample-stages -rostype interpolate -sdim 1000 -writesol
    $MERGE -nset 1 -romsns -romos -rostype interpolate -nwinsamp 10 -eqp -nwinover 4
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.02 -online -romhrprep -romsns -nwin 3 -rostype interpolate -hrsamptype eqp -maxnnls 100
    $LAGHOS -m data/cube01_hex.mesh -rs 1 -pt 211 -tf 0.02 -online -romhr -romsns -nwin 3 -rostype interpolate -hrsamptype eqp
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -rs 1 -pt 211 -restore -nwin 3 -romsns -soldiff -rostype interpolate -hrsamptype eqp -tf 0.02
    ;;
esac
