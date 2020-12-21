NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7"
absoluteFOMTol="1e-7"
absoluteFOMTolParallel="1e-7"
absoluteRelErrorTol="3.0e-1"
absoluteRelErrorTolParallel="3.0e-1"
speedupTol="0.8"
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -offline -writesol -ef 0.9999 -romsvds -nwin 4 -tw "$BASE_DIR"/tests/gresho-vortices/absolute-gresho-vortices-time-window.csv -sdim 800 -rostype load
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62 -s 7 -online -romhr -soldiff -romsvds -nwin 4 -twp twpTemp.csv -sfacx 35 -sfacv 35 -sface 35 -rostype load
    ;;
esac
