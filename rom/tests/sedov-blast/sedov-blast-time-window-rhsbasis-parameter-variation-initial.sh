NUM_PARALLEL_PROCESSORS=8
testNames=(fom romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -offline -romsvds -romos -rostype initial -romsrhs -bef 1.0 -rpar 0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -writesol
    $MERGE -nset 1 -romos -rostype initial -rhs -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -twp twpTemp.csv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhrprep -romos -rostype initial -sfacx 1 -sfacv 32 -sface 32 -romgs -romsrhs -bef 1.0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -twp twpTemp.csv
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhr -romos -rostype initial -sfacx 1 -sfacv 32 -sface 32 -romgs -romsrhs -bef 1.0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -twp twpTemp.csv
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 2 -soldiff -romos -rostype initial
    ;;
esac
