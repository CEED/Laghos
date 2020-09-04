NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.1 -offline -romsvds -romos -romsrhs -bef 1.0 -rpar 0 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -writesol -visit
    $HEADER ./merge -nset 1 -rhs -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv
    ;;
  2)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.1 -online -romhr -romos -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0 -rpar 1 -nwin 2 -tw "$BASE_DIR"/tests/sedov-blast/sedov-blast-time-window-rhsbasis-parameter-variation.csv -twp twpTemp.csv
    ;;
esac
