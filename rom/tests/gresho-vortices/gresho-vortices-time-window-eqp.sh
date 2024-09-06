NUM_PARALLEL_PROCESSORS=1
testNames=(offline online restore)
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.02 -s 7 -offline -writesol -romsns -sample-stages -rostype interpolate -sdim 10000 -rpar 0 -cfl 0.35
    $MERGE -nset 1 -romsns -romos -rostype interpolate -nwinsamp 10 -eqp
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.02 -s 7 -online -romhrprep -romsns -nwin 2 -rostype interpolate -hrsamptype eqp -cfl 0.35 -lqnnls
    ;;
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.02 -s 7 -online -romhr -romsns -nwin 2 -rostype interpolate -hrsamptype eqp -cfl 0.35 -lqnnls
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.02 -s 7 -restore -soldiff -romsns -nwin 2 -rostype interpolate -cfl 0.35
    ;;
esac
