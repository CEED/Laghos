NUM_PARALLEL_PROCESSORS=1
testNames=(offline online restore)
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -tf 0.04 -s 7 -rs 1 -cfl 0.05 -pa -offline -writesol -romsns -sample-stages -rostype interpolate -sdim 10000 -rpar 0
    $MERGE -nset 1 -romsns -romos -rostype interpolate -nwinsamp 10 -eqp
    ;;
  2)
    $LAGHOS -p 3 -m data/box01_hex.mesh -tf 0.04 -s 7 -rs 1 -cfl 0.05 -pa -online -romhrprep -romsns -nwin 4 -rostype interpolate -hrsamptype eqp -lqnnls -maxnnls 100
    $LAGHOS -p 3 -m data/box01_hex.mesh -tf 0.04 -s 7 -rs 1 -cfl 0.05 -pa -online -romhr -romsns -nwin 4 -rostype interpolate -hrsamptype eqp
    ;;
  3)
    $LAGHOS -p 3 -m data/box01_hex.mesh -tf 0.04 -s 7 -rs 1 -cfl 0.05 -pa -restore -soldiff -romsns -nwin 4 -rostype interpolate
    ;;
esac
