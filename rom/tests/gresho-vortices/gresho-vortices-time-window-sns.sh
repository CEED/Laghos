NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -offline -ef 0.9999 -writesol -romsvds -nwin 4 -tw "$BASE_DIR"/tests/gresho-vortices/gresho-vortices-time-window.csv -sdim 800 -romsns -romos
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -online -romgs -soldiff -romsvds -nwin 4 -romsns -romos
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -online -romgs -romhrprep -romsvds -sfacx 50 -sfacv 50 -sface 50 -nwin 4 -romsns -romos
    $LAGHOS_SERIAL -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -online -romgs -romhr -romsvds -sfacx 50 -sfacv 50 -sface 50 -nwin 4 -romsns -romos
    ;;
  4)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -restore -nwin 4 -soldiff -romsns -romos
    ;;
esac