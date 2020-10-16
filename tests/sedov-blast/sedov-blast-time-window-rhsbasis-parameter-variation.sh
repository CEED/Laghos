NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
runAbsoluteFOM="false"
absoluteFOMOptions="-m data/cube01_hex.mesh -pt 211 -tf 0.1"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.1 -offline -romsvds -romos -romsrhs -bef 1.0 -rpar 0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -writesol -visit
    $MERGE -nset 1 -rhs -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.1 -online -romhr -romos -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0 -rparos -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -twp twpTemp.csv
    ;;
esac
