NUM_PARALLEL_PROCESSORS=4
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -offline -ef 0.9999 -writesol -romsvds -romos -romsrhs -nwinsamp 60
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -nwin 2 -twp twpTemp.csv -romgs
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -sfacx 10 -sfacv 10 -sface 10 -soldiff -romsrhs -nwin 2 -twp twpTemp.csv -romgs -romhr
    ;;
  4)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -nwin 2 -twp twpTemp.csv -romsrhs
    ;;
esac
