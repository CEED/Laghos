NUM_PARALLEL_PROCESSORS=4
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -offline -ef 0.9999 -writesol -romsvds -romos -rostype load -romsrhs -nwinsamp 20 -twp twpTemp.csv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -nwin 2 -twp twpTemp.csv -romgs
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -romsrhs -nwin 2 -twp twpTemp.csv -romgs -romhrprep
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.025 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -romsrhs -nwin 2 -twp twpTemp.csv -romgs -romhr
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 2 -twp twpTemp.csv -romsrhs -soldiff -romos -rostype load
    ;;
esac
