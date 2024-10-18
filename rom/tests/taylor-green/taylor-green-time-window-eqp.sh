NUM_PARALLEL_PROCESSORS=1
testNames=(offline online restore)
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 2 -m data/cube01_hex.mesh -cfl 0.1 -tf 0.005 -s 7 -offline -writesol -romsns -sample-stages -rostype interpolate -sdim 1000 -rpar 0
    $MERGE -nset 1 -romsns -romos -rostype interpolate -nwinsamp 10 -eqp
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -cfl 0.1 -p 0 -s 7 -tf 0.005 -online -romhrprep -romsns -nwin 3 -rostype interpolate -hrsamptype eqp -lqnnls -maxnnls 100
    $LAGHOS -m data/cube01_hex.mesh -cfl 0.1 -p 0 -s 7 -tf 0.005 -online -romhr -romsns -nwin 3 -rostype interpolate -hrsamptype eqp
    ;;
  3)
    $LAGHOS -p 0 -m data/cube01_hex.mesh -cfl 0.1 -tf 0.005 -s 7 -restore -soldiff -romsns -nwin 3 -rostype interpolate
    ;;
esac
