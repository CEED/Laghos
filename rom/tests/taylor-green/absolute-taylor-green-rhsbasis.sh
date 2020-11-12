NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-p 0 -rs 2 -iv -cfl 0.5 -tf 0.25 -pa"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.25 -pa -offline -romsvds -ef 0.9999 -writesol -romos -romsrhs -nwinsamp 10 -rostype load
    ;;
  2)
    $LAGHOS -p 0 -rs 2 -iv -cfl 0.5 -tf 0.25 -pa -online -soldiff -romsvds -romos -romhr -romsrhs -romgs -sfacv 128 -sface 128 -twp twpTemp.csv -nwin 35 -rostype load
    ;;
esac
