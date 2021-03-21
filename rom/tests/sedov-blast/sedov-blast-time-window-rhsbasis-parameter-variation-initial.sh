NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -offline -romsvds -romos -rostype initial -bef 1.0 -rpar 0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -writesol
    $MERGE -nset 1 -romos -rostype initial -rhs -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhrprep -romos -rostype initial -sfacx 1 -sfacv 32 -sface 32 -romgs -bef 1.0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhr -romos -rostype initial -sfacx 1 -sfacv 32 -sface 32 -romgs -bef 1.0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv
    ;;
esac
