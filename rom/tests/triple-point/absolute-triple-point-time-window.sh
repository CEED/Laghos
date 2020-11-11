NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -offline -writesol -romsvds -ef 0.9999 -nwin 4 -tw "$BASE_DIR"/tests/triple-point/triple-point-time-window.csv
    ;;
  2)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -soldiff -nwin 4 -romhr -twp twpTemp.csv
    ;;
esac
