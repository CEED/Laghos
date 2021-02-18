NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -offline -romsvds -romos -rostype interpolate -romsrhs -bef 1.0 -rpar 0 -writesol
    $MERGE -nset 1 -romos -rostype interpolate -rhs -nwinsamp 20 -nwinover 5
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhrprep -romos -rostype interpolate -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0 -nwin 2 -twp twpTemp.csv
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romhr -romos -rostype interpolate -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0 -nwin 2 -twp twpTemp.csv
    ;;
esac