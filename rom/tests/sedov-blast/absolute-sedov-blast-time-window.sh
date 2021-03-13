NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-m data/cube01_hex.mesh -pt 211 -tf 0.4"
absoluteFOMTol="1e-7"
absoluteFOMTolParallel="1e-7"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="2"
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.4 -offline -ef 0.999999 -nwin 4 -tw "$BASE_DIR"/tests/sedov-blast/absolute-sedov-blast-time-window.csv -writesol -romsvds -rostype load
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.4 -online -romhrprep -nwin 4 -twp twpTemp.csv -soldiff -rostype load
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.4 -online -romhr -nwin 4 -twp twpTemp.csv -soldiff -rostype load
    ;;
esac
