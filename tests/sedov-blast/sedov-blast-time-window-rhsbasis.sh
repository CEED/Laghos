NUM_PARALLEL_PROCESSORS=4
testNames=(offline online romhr restore)
runAbsoluteFOM="true"
absoluteFOMOptions="-m data/cube01_hex.mesh -pt 211 -tf 0.05"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -offline -ef 0.9999 -writesol -romsvds -romos -rostype load -romsrhs -nwinsamp 60
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -nwin 2 -twp twpTemp.csv -romgs
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -nwin 2 -twp twpTemp.csv -romgs -romhr
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 2 -twp twpTemp.csv -romsrhs
    ;;
esac
