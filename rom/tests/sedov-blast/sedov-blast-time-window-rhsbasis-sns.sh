NUM_PARALLEL_PROCESSORS=4
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -offline -ef 0.9999 -writesol -romsvds -romos -rostype load -nwinsamp 20 -romsns
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -soldiff -nwin 2 -romsns
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -nwin 2 -romsns -romhrprep
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -nwin 2 -romsns -romhr
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 2 -soldiff -romos -rostype load -romsns
    ;;
esac
