NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.1 -offline -romsvds -romos -rostype previous -romsrhs -bef 1.0 -rpar 0 -writesol -visit
    $MERGE -nset 1 -romos -rostype previous -rhs -nwinsamp 125 -nwinover 30
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.1 -online -romhr -romos -rostype previous -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0 -twp twpTemp.csv
    ;;
esac
