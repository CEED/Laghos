NUM_PARALLEL_PROCESSORS=8
testNames=(fom romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -romsvds -romos -rostype interpolate -bef 1.0 -rpar 0 -romsns
    $MERGE -nset 1 -romos -rostype interpolate -romsns
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -bef 0.5 -writesol
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 7 -rdimv 12 -rdime 6 -romhrprep -romos -rostype interpolate -sfacx 1 -sfacv 32 -sface 32 -romsns -bef 0.5
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 7 -rdimv 12 -rdime 6 -romhr -romos -rostype interpolate -sfacx 1 -sfacv 32 -sface 32 -romsns -bef 0.5
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -soldiff -rdimx 7 -rdimv 12 -rdime 6 -romos -rostype interpolate -romsns
    ;;
esac
